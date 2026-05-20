import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, MultiHeadTokenMixing, PerTokenSwiGLU, SwiGLU
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class MixFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="MixFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
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
        super(MixFormer, self).__init__(feature_map,
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
        # 非序列特征 tokenizer: 产生 num_ns_token 个 NS tokens，因为target item 也被视为一个NS token，所以加入了self.item_info_dim
        self.tokenizer_layer = Embedding2Tokenization(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=token_dim,
            num_ns_token=num_ns_token
        )

        # item sequence / target item 投影到统一 token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.unified_layers = MixFormerBlocks(input_dim=token_dim,
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

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x T x token_dim

        # target item 作为一个 NS token
        # 其它非序列特征 -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # B x non_item_dim
        batch_emb = torch.cat([batch_emb, target_emb], dim=-1)
        ns_tokens = self.tokenizer_layer(batch_emb)  # B x num_ns_token x token_dim

        # unified model
        _, ns_tokens = self.unified_layers(s_tokens, ns_tokens, mask)

        bottom_output = ns_tokens.mean(dim=1)
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

class MixFormerBlocks(nn.Module):
    """
    每一层依次执行：
    1) Query Mixer
    2) Cross Attention
    3) Output Fusion
    """
    def __init__(self,
                 input_dim,
                 num_ns_token,
                 num_layers,
                 expand=4,
                 net_dropout=0.0):
        super(MixFormerBlocks, self).__init__()
        self.num_layers = num_layers

        self.query_mixers = nn.ModuleList([
            QueryMixer(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

        self.cross_attentions = nn.ModuleList([
            CrossAttention(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

        self.output_fusions = nn.ModuleList([
            OutputFusion(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x T x D
        ns_tokens: B x N x D
        mask:      B x T
        """
        for i in range(self.num_layers):
            ns_tokens = self.query_mixers[i](ns_tokens)                 # B x N x D
            s_tokens, ns_tokens = self.cross_attentions[i](s_tokens, ns_tokens, mask)
            ns_tokens = self.output_fusions[i](ns_tokens)               # B x N x D
        return s_tokens, ns_tokens

class QueryMixer(nn.Module):
    """
    对应论文中的 Query Mixer:
    P = HeadMixing(Norm(X)) + X
    q_i = PerHeadFFN(Norm(p_i)) + p_i
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(QueryMixer, self).__init__()
        if hasattr(nn, "RMSNorm"):
            self.norm1 = nn.RMSNorm(input_dim)
            self.norm2 = nn.RMSNorm(input_dim)
        else:
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
        self.head_mixing = MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token)
        self.per_head_ffn = PerTokenSwiGLU(input_dim=input_dim,
                                                num_token=num_ns_token,
                                                expand=expand,
                                                net_dropout=net_dropout)

    def forward(self, x):
        x = self.head_mixing(self.norm1(x)) + x
        x = self.per_head_ffn(self.norm2(x)) + x
        return x

class CrossAttention(nn.Module):
    """
    对应论文中的 Cross Attention:
    - 先对序列做 per-layer FFN refinement
    - 再由每个 NS head 对序列做 cross attention
    - trade-off：没有直接在scaled_dot_product_attention引入自定义mask，用少量的注意力权重稀释换取FlashAttention的计算加速，
      如果在scaled_dot_product_attention直接引入自定义mask，反而效果降低
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.num_ns_token = num_ns_token

        if hasattr(nn, "RMSNorm"):
            self.seq_norm = nn.RMSNorm(input_dim)
        else:
            self.seq_norm = nn.LayerNorm(input_dim)

        self.seq_ffn = SwiGLU(
            input_dim=input_dim,
            expand=expand,
            net_dropout=net_dropout
        )

        # 每个 query head 一组 K/V 投影
        self.k_proj = nn.Linear(input_dim, input_dim * num_ns_token)
        self.v_proj = nn.Linear(input_dim, input_dim * num_ns_token)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens: B x T x D
        ns_tokens: B x N x D
        mask:     B x T (1/True 表示有效位置)
        """
        # per-layer sequence refinement
        s_tokens = self.seq_ffn(self.seq_norm(s_tokens)) + s_tokens  # B x T x D

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        keys = self.k_proj(s_tokens).chunk(self.num_ns_token, dim=-1)
        values = self.v_proj(s_tokens).chunk(self.num_ns_token, dim=-1)
        k = torch.stack(keys, dim=1) # K/V: B x N x T x D
        v = torch.stack(values, dim=1)  # K/V: B x N x T x D
        q = ns_tokens.unsqueeze(2) # B × N x 1 x D

        # B x N x D
        ns_tokens = F.scaled_dot_product_attention(q, k, v).squeeze(2) + ns_tokens
        return s_tokens, ns_tokens

class OutputFusion(nn.Module):
    """
    对应论文中的 Output Fusion:
    o_i = PerHeadFFN(Norm(z_i)) + z_i
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(OutputFusion, self).__init__()
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(input_dim)
        else:
            self.norm = nn.LayerNorm(input_dim)
        self.per_head_ffn = PerTokenSwiGLU(input_dim=input_dim,
                                           num_token=num_ns_token,
                                           expand=expand,
                                           net_dropout=net_dropout)

    def forward(self, ns_tokens):
        ns_tokens = self.per_head_ffn(self.norm(ns_tokens)) + ns_tokens
        return ns_tokens