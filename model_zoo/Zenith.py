import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, DIN_Attention, PerTokenSwiGLU
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class Zenith(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="Zenith",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 din_use_softmax=False,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 id_dim=64,
                 token_dim=64,
                 num_token=4,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(Zenith, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        if isinstance(tower_activations, str) and tower_activations.lower() == "dice":
            tower_activations = [Dice(units) for units in tower_hidden_units]
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_token = num_token
        self.accumulation_steps = accumulation_steps
        self.id_dim = id_dim

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
                if feat in ["item_id"]:
                    self.item_info_dim += id_dim
                else:
                    self.item_info_dim += emb_dim
            else:
                if feat in ["user_id"]:
                    continue
                self.non_item_dim += emb_dim

        self.user_id_embedding_layer = FeatureEmbedding(feature_map, id_dim, required_feature_columns=['user_id'])
        self.item_id_embedding_layer = FeatureEmbedding(feature_map, id_dim, required_feature_columns=['item_id'])

        self.feature_embedding_layer = FeatureEmbedding(
            feature_map, embedding_dim,
            not_required_feature_columns=['user_id', 'item_id']
        )

        user_profile_dim = self.non_item_dim
        item_attribute_dim = self.item_info_dim * 2
        self.tokenizer_layer = Embedding2Tokenization(
            id_dim,
            user_profile_dim,
            item_attribute_dim,
            token_dim,
            num_token
        )

        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )

        # 最终 token 数 = user_id(1) + item_id(1) + user_profile_tokens(num_ns) + item_attribute_tokens(num_ns)
        self.num_prime_tokens = 2 + 2 * num_token

        self.unified_layers = ZenithBlock(input_dim=token_dim,
                                         num_token=self.num_prime_tokens,
                                         num_layers=num_layers,
                                         expand=expansion_factor,
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
        # item_dict 中假设包含 [history_items..., target_item]
        # flatten_emb=True 后再 reshape 成: B x (T+1) x item_info_dim
        item_attri_emb = self.feature_embedding_layer(item_dict, flatten_emb=True)
        item_attri_emb = item_attri_emb.view(batch_size, -1, self.item_info_dim - self.id_dim)
        item_id_emb = self.item_id_embedding_layer(item_dict, flatten_emb=True)
        item_id_emb = item_id_emb.view(batch_size, -1, self.id_dim)

        item_feat_emb = torch.cat([item_attri_emb, item_id_emb], dim=-1)
        target_emb = item_feat_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x T x item_info_dim

        # 其它非序列特征 -> NS tokens
        user_profile = self.feature_embedding_layer(batch_dict, flatten_emb=True)
        user_id_emb = self.user_id_embedding_layer(batch_dict, flatten_emb=True)
        item_attribute = self.attention_layers(target_emb, sequence_emb, mask)
        item_attribute = torch.cat([target_emb, item_attribute], dim=-1)
        tokens = self.tokenizer_layer(user_id_emb.unsqueeze(1), item_id_emb[:, -1, :].unsqueeze(1), user_profile, item_attribute)

        # unified model
        tokens = self.unified_layers(tokens)

        bottom_output = tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class Embedding2Tokenization(nn.Module):
    def __init__(self, id_dim, user_profile_dim, item_attribute_dim, token_dim, num_token):
        super(Embedding2Tokenization, self).__init__()
        self.num_token = num_token
        self.token_dim = token_dim
        output_dim = token_dim * num_token

        self.user_profile_mlp = nn.Sequential(
            nn.Linear(user_profile_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.item_attribute_mlp = nn.Sequential(
            nn.Linear(item_attribute_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.user_id_mlp = nn.Sequential(
            nn.Linear(id_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim)
        )
        self.item_id_mlp = nn.Sequential(
            nn.Linear(id_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim)
        )


    def forward(self, user_id_emb, item_id_emb, user_profile, item_attribute):
        """
        user_id_token: [B, 1, D]
        item_id_token: [B, 1, D]
        user_profile: [B, user_profile_dim]
        item_attribute: [B, item_attribute_dim]
        return: [B, 2 + 2*num_token, token_dim]
        """
        user_profile = self.user_profile_mlp(user_profile)
        user_profile_tokens = user_profile.view(-1, self.num_token, self.token_dim)

        item_attribute = self.item_attribute_mlp(item_attribute)
        item_attribute_tokens = item_attribute.view(-1, self.num_token, self.token_dim)

        user_id_token = self.user_id_mlp(user_id_emb)
        item_id_token = self.item_id_mlp(item_id_emb)

        tokens = torch.cat([user_id_token, item_id_token, user_profile_tokens, item_attribute_tokens], dim=1)
        return tokens

class ZenithBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 num_layers,
                 num_head=2,
                 expand=2,
                 net_dropout=0.0):
        super(ZenithBlock, self).__init__()
        self.num_layers = num_layers

        if hasattr(nn, "RMSNorm"):
            self.mixer_norms = nn.ModuleList([
                nn.RMSNorm(input_dim)
                for _ in range(num_layers)
            ])
            self.pffn_norms = nn.ModuleList([
                nn.RMSNorm(input_dim)
                for _ in range(num_layers)
            ])
        else:
            self.mixer_norms = nn.ModuleList([
                nn.LayerNorm(input_dim)
                for _ in range(num_layers)
            ])
            self.pffn_norms = nn.ModuleList([
                nn.LayerNorm(input_dim)
                for _ in range(num_layers)
            ])

        self.mixer_layers = nn.ModuleList([
            TokenwiseMultiHeadSelfAttention(token_dim=input_dim,
                                            num_token=num_token,
                                            num_head=num_head)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenSwiGLU(input_dim=input_dim,
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

class TokenwiseMultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_head=2):
        super(TokenwiseMultiHeadSelfAttention, self).__init__()
        assert token_dim % num_head == 0, "token_dim must be divisible by num_head"
        self.num_head = num_head
        self.head_dim = token_dim // num_head
        self.token_dim = token_dim
        self.num_token = num_token
        self.W_q = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_k = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_v = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_o = nn.Linear(token_dim, token_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, tokens): # B × T × D
        Q = torch.einsum("btd,tdh->bth", tokens, self.W_q)
        K = torch.einsum("btd,tdh->bth", tokens, self.W_k)
        V = torch.einsum("btd,tdh->bth", tokens, self.W_v)

        Q = Q.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2) # [B, H, T, Dh]
        K = K.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, self.num_token, self.token_dim)  # [B, T, D]
        attn_out = self.W_o(attn_out)
        return attn_out