import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, DIN_Attention, PerTokenFeedForward, MultiHeadTokenMixing
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class RankMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
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
                 token_dim=64,
                 num_ns_token=4,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(RankMixer, self).__init__(feature_map,
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
        self.num_ns_token = num_ns_token
        self.accumulation_steps = accumulation_steps

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
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.tokenizer_layer = Embedding2Tokenization(
            input_dim=input_dim,
            token_dim=token_dim,
            num_ns_token=num_ns_token
        )

        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )
        self.unified_layers = RankMixerBlock(input_dim=token_dim,
                                         num_ns_token=self.num_ns_token,
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
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x T x item_info_dim

        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask) # B x embedding_dim

        # 其它非序列特征 -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        context_tokens = torch.cat([batch_emb, target_emb, pooling_emb], dim=-1)
        context_tokens = self.tokenizer_layer(context_tokens)                     # B x num_ns_token x token_dim


        # unified model
        context_tokens = self.unified_layers(context_tokens)

        bottom_output = context_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class Embedding2Tokenization(nn.Module):
    def __init__(self, input_dim, token_dim, num_ns_token):
        super(Embedding2Tokenization, self).__init__()
        self.num_ns_token = num_ns_token
        self.token_dim = token_dim
        output_dim = token_dim * num_ns_token
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, feature_embeddings):
        """
        feature_embeddings: B x input_dim
        return: B x num_ns_token x token_dim
        """
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
        flatten_ns_tokens = self.mlp(feature_embeddings)
        ns_tokens = flatten_ns_tokens.view(-1, self.num_ns_token, self.token_dim)
        return ns_tokens

class RankMixerBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_ns_token,
                 num_layers,
                 expand=2,
                 net_dropout=0.0):
        super(RankMixerBlock, self).__init__()
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
            MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(input_dim=input_dim,
                           num_token=num_ns_token,
                           expand=expand,
                           net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x