import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedAveragePooling, PerTokenFeedForward, DIN_Attention
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class HiFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="HiFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=3,
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 din_use_softmax=False,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_heads=2,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(HiFormer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.num_field = feature_map.get_num_fields()
        # 统计非 item 特征维度、item 特征维度
        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.num_field += 1
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.tokenizer_layer = Embedding2Tokenization(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_field=self.num_field,
            num_tasks=num_tasks
        )

        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )

        self.unified_layers = HiFormerBlocks(token_dim=token_dim,
                                       num_token=num_tasks + self.num_field,
                                       num_layers=num_layers,
                                       expand=expansion_factor,
                                       num_heads=num_heads,
                                       net_dropout=net_dropout)
        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim,
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=tower_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_tasks)])
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x T x item_info_dim

        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)

        # 其它非序列特征 -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=False)
        context_tokens = torch.cat([batch_emb,
                                    target_emb.view(batch_size, -1, self.embedding_dim),
                                    pooling_emb.view(batch_size, -1, self.embedding_dim)], dim=1)
        context_tokens = self.tokenizer_layer(context_tokens)

        # unified model
        context_tokens = self.unified_layers(context_tokens)

        # use task tokens for task-specific towers
        task_tokens = context_tokens[:, self.num_field:, :]  # [B, num_tasks, token_dim]
        tower_output = [self.tower[i](task_tokens[:, i, :]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class Embedding2Tokenization(nn.Module):
    def __init__(self, embedding_dim, token_dim, num_field, num_tasks):
        super(Embedding2Tokenization, self).__init__()
        self.token_dim = token_dim
        # --------------- learnable task tokens (like CLS in BERT) ---------------
        self.task_tokens = nn.init.xavier_uniform_(nn.Parameter(torch.empty(1, num_tasks, token_dim)))
        self.W = nn.init.xavier_uniform_(nn.Parameter(torch.empty(num_field, embedding_dim, token_dim)))

    def forward(self, feature_embeddings):
        """
        feature_embeddings: B x F x emb_dim
        return: B x T x D
        """
        tokens = torch.einsum("bfd,fdk->bfk", feature_embeddings, self.W)
        task_tokens = self.task_tokens.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([tokens, task_tokens], dim=1)
        return tokens


class HiFormerBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_layers,
                 expand=2,
                 num_heads=2,
                 net_dropout=0.0):
        super(HiFormerBlocks, self).__init__()
        self.num_layers = num_layers
        if hasattr(nn, "RMSNorm"):
            self.mixer_norms = nn.ModuleList([
                nn.RMSNorm(token_dim)
                for _ in range(num_layers)
            ])
            self.pffn_norms = nn.ModuleList([
                nn.RMSNorm(token_dim)
                for _ in range(num_layers)
            ])
        else:
            self.mixer_norms = nn.ModuleList([
                nn.LayerNorm(token_dim)
                for _ in range(num_layers)
            ])
            self.pffn_norms = nn.ModuleList([
                nn.LayerNorm(token_dim)
                for _ in range(num_layers)
            ])

        self.mixer_layers = nn.ModuleList([
            HiformerAttentionLayer(token_dim=token_dim,
                                        num_token=num_token,
                                        num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(input_dim=token_dim,
                                num_token=num_token,
                                expand=expand,
                                net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x

class HiformerAttentionLayer(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_heads=2):
        super(HiformerAttentionLayer, self).__init__()
        assert token_dim % num_heads == 0, "token_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.token_dim = token_dim
        self.num_token = num_token
        input_dim = num_token * token_dim
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor): # B × T × D
        x = x.flatten(start_dim=1) # B × TD
        Q = self.W_q(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T, Dh]
        K = self.W_k(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, self.num_token, self.token_dim)  # [B, T, D]
        return attn_out