import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, MultiHeadTokenMixing, PerTokenSwiGLU, SwiGLU
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class EST(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="EST",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 dnn_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=3,
                 num_heads=2,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(EST, self).__init__(feature_map,
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
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_ns_token = self.num_field
        self.tokenizer_layer = Embedding2Tokenization(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_field=self.num_field
        )

        # item sequence / target item 投影到统一 token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.unified_layers = ESTBlocks(
            input_dim=token_dim,
            num_heads=num_heads,
            num_ns_token=self.num_ns_token,
            num_layers=num_layers,
            dnn_activations=dnn_activations,
            expansion_factor=expansion_factor
        )

        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim * self.num_field + token_dim,
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

        target_emb = item_feat_emb[:, -1, :].view(batch_size, -1, self.embedding_dim)
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x T x item_info_dim

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x T x token_dim

        # 其它非序列特征 -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=False)  # B x
        feature_emb = torch.cat([batch_emb, target_emb], dim=1) # B x F x embeddding_dim
        ns_tokens = self.tokenizer_layer(feature_emb)  # B x num_ns_token (F) x token_dim

        # unified model
        s_tokens, ns_tokens = self.unified_layers(s_tokens, ns_tokens, mask)

        bottom_output = torch.cat([ns_tokens.flatten(start_dim=1), s_tokens.mean(dim=1)], dim=-1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class Embedding2Tokenization(nn.Module):
    def __init__(self, embedding_dim, token_dim, num_field):
        super(Embedding2Tokenization, self).__init__()
        self.token_dim = token_dim
        self.W = nn.Parameter(torch.empty(num_field, embedding_dim, token_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feature_embeddings):
        """
        feature_embeddings: B x F x emb_dim
        return: B x T x D
        """
        tokens = torch.einsum("bfd,fdk->bfk", feature_embeddings, self.W)
        return tokens

class ESTBlocks(nn.Module):
    """
    每一层依次执行：
    1) Lightweight Cross-Attention (由于没有多模态表征，去除Content Sparse Attention)
    2) PerToken FFN + Shared FFN
    """
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 dnn_activations='ReLU',
                 expansion_factor=4):
        super(ESTBlocks, self).__init__()
        self.num_layers = num_layers

        if hasattr(nn, "RMSNorm"):
            self.attention_norms = nn.ModuleList([
                nn.RMSNorm(input_dim) for _ in range(num_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.RMSNorm(input_dim) for _ in range(num_layers)
            ])
        else:
            self.attention_norms = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers)
            ])

        self.lca_layers = nn.ModuleList([
            LightweightCrossAttention(
                input_dim=input_dim,
                num_heads=num_heads,
                num_ns_token=num_ns_token
            ) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            MixedFFN(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                dnn_activations=dnn_activations,
                expansion_factor=expansion_factor
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        for i in range(self.num_layers):
            if mask is not None:
                s_tokens = s_tokens * mask.unsqueeze(-1).float()

            norm_s = self.attention_norms[i](s_tokens)
            norm_ns = self.attention_norms[i](ns_tokens)
            ns_tokens = self.lca_layers[i](norm_s, norm_ns, mask) + ns_tokens

            norm_s = self.ffn_norms[i](s_tokens)
            norm_ns = self.ffn_norms[i](ns_tokens)
            norm_s, norm_ns = self.ffn_layers[i](norm_s, norm_ns, mask)
            s_tokens = s_tokens + norm_s
            ns_tokens = ns_tokens + norm_ns

            if mask is not None:
                s_tokens = s_tokens * mask.unsqueeze(-1).float()

        return s_tokens, ns_tokens


class LightweightCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_ns_token):
        super(LightweightCrossAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads

        # shared projections for S-tokens
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)

        # token-specific projections for NS-tokens
        # shape: [num_ns_token, input_dim, input_dim]
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x Ls x D
        ns_tokens: B x Lns x D
        mask:      B x Ls, 1 for valid positions
        """
        B, Ls, D = s_tokens.shape
        _, Lns, _ = ns_tokens.shape

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        # shared KV for S-tokens
        k_s = self.W_k_s(s_tokens)  # B x Ls x D
        v_s = self.W_v_s(s_tokens)  # B x Ls x D

        # token-specific QKV for NS-tokens using einsum
        # ns_tokens: B x Lns x D
        # output:    B x Lns x D
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)

        # split heads
        q = q_ns.view(B, Lns, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x Lns x h
        k = k_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2) # B x H x Ls x h
        v = v_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2) # B x H x Ls x h

        # unified causal attention
        output = F.scaled_dot_product_attention(q, k, v) # B x H x Lns x h

        # concat heads
        output = output.transpose(1, 2).contiguous().view(B, Lns, D)
        output = torch.einsum("bld,ldh->blh", output, self.W_o)

        return output


class MixedFFN(nn.Module):
    def __init__(self, input_dim, num_ns_token, dnn_activations, expansion_factor=4):
        super(MixedFFN, self).__init__()
        hidden_dim = input_dim * expansion_factor
        self.num_ns_token = num_ns_token
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = get_activation(dnn_activations)

        # shared FFN for S-tokens
        self.ffn_s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(dnn_activations),
            nn.Linear(hidden_dim, input_dim)
        )

        # token-specific FFN for NS-tokens using einsum-style parameters
        self.W1_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, hidden_dim))
        self.b1_ns = nn.Parameter(torch.zeros(num_ns_token, hidden_dim))
        self.W2_ns = nn.Parameter(torch.empty(num_ns_token, hidden_dim, input_dim))
        self.b2_ns = nn.Parameter(torch.zeros(num_ns_token, input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1_ns)
        nn.init.xavier_uniform_(self.W2_ns)
        nn.init.zeros_(self.b1_ns)
        nn.init.zeros_(self.b2_ns)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x Ls x D
        ns_tokens: B x Lns x D
        """
        # shared FFN for S-tokens
        s_out = self.ffn_s(s_tokens)

        # token-specific FFN for NS-tokens
        ns_hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1_ns) + self.b1_ns.unsqueeze(0)
        ns_hidden = self.activation(ns_hidden)
        ns_out = torch.einsum("blh,lhd->bld", ns_hidden, self.W2_ns) + self.b2_ns.unsqueeze(0)

        if mask is not None:
            s_out = s_out * mask.unsqueeze(-1).float()

        return s_out, ns_out