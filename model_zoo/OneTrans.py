import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class OneTrans(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="OneTrans",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(OneTrans, self).__init__(feature_map,
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

        # OneTrans 主体
        self.unified_layers = OneTransBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=num_ns_token,
            dnn_activations=dnn_activations,
            expansion_factor=expansion_factor
        )

        # 最终只用 NS tokens 做预测
        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim * num_ns_token,
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
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x S x item_info_dim

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x S x token_dim

        # target item 作为一个 NS token
        # 其它非序列特征 -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        batch_emb = torch.cat([batch_emb, target_emb], dim=-1)
        ns_tokens = self.tokenizer_layer(batch_emb)                     # B x num_ns_token x token_dim

        # unified OneTrans
        ns_tokens = self.unified_layers(s_tokens, ns_tokens, mask)

        # 最终使用 NS tokens 做预测
        bottom_output = ns_tokens.flatten(start_dim=1)
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

class OneTransBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 dnn_activations='ReLU',
                 expansion_factor=4):
        super(OneTransBlock, self).__init__()
        self.num_layers = num_layers

        if hasattr(nn, "RMSNorm"):
            self.norm1_layers = nn.ModuleList([
                nn.RMSNorm(input_dim) for _ in range(num_layers)
            ])
            self.norm2_layers = nn.ModuleList([
                nn.RMSNorm(input_dim) for _ in range(num_layers)
            ])
        else:
            self.norm1_layers = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers)
            ])
            self.norm2_layers = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers)
            ])

        self.mha_layers = nn.ModuleList([
            MixedMHA(
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
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()
            q_mask = mask
        else:
            q_mask = None

        ps_tokens = s_tokens
        for i in range(self.num_layers):
            # sequence query pruning
            start = ps_tokens.size(1) // 2
            ps_tokens = ps_tokens[:, start:, :]
            if q_mask is not None:
                q_mask = q_mask[:, start:]
            norm_s = self.norm1_layers[i](s_tokens)
            norm_ps = self.norm1_layers[i](ps_tokens)
            norm_ns = self.norm1_layers[i](ns_tokens)

            delta_ps, delta_ns = self.mha_layers[i](norm_s, norm_ps, norm_ns, kv_mask=mask, q_mask=q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            norm_ps = self.norm2_layers[i](ps_tokens)
            norm_ns = self.norm2_layers[i](ns_tokens)

            delta_ps, delta_ns = self.ffn_layers[i](norm_ps, norm_ns, q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            s_tokens = ps_tokens
            mask = q_mask
        return ns_tokens


class MixedMHA(nn.Module):
    def __init__(self, input_dim, num_heads, num_ns_token):
        super(MixedMHA, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads

        # shared projections for S-tokens
        self.W_q_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)

        # token-specific projections for NS-tokens
        # shape: [num_ns_token, input_dim, input_dim]
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)

    def build_mask(self, B, Ls, qLs, Lns, device, kv_mask=None, q_mask=None):
        """
        Args:
            B: batch size
            Ls: full S length
            qLs: pruned S query length
            Lns: NS length
            kv_mask: [B, Ls], valid mask for full S key/value
            q_mask:  [B, qLs], valid mask for pruned S query

        Returns:
            attn_mask: [B, 1, Lq, L], bool
        """
        if kv_mask is None:
            full_k_mask = torch.ones(B, Ls, dtype=torch.bool, device=device)
        else:
            full_k_mask = kv_mask.bool()

        if q_mask is None:
            pruned_q_mask = torch.ones(B, qLs, dtype=torch.bool, device=device)
        else:
            pruned_q_mask = q_mask.bool()

        ns_k_mask = torch.ones(B, Lns, dtype=torch.bool, device=device)
        ns_q_mask = torch.ones(B, Lns, dtype=torch.bool, device=device)

        key_valid = torch.cat([full_k_mask, ns_k_mask], dim=1)      # [B, Ls + Lns]
        query_valid = torch.cat([pruned_q_mask, ns_q_mask], dim=1)  # [B, qLs + Lns]

        q_start = Ls - qLs

        s_q_pos = torch.arange(q_start, Ls, device=device)              # [qLs]
        ns_q_pos = torch.arange(Ls, Ls + Lns, device=device)           # [Lns]
        q_pos = torch.cat([s_q_pos, ns_q_pos], dim=0)           # [Lq]
        q_pos = q_pos.unsqueeze(0).expand(B, -1)                       # [B, Lq]

        s_k_pos = torch.arange(Ls, device=device)                      # [Ls]
        ns_k_pos = torch.arange(Ls, Ls + Lns, device=device)           # [Lns]
        k_pos = torch.cat([s_k_pos, ns_k_pos], dim=0)           # [L]
        k_pos = k_pos.unsqueeze(0).expand(B, -1)                       # [B, L]

        causal_mask = k_pos.unsqueeze(1) <= q_pos.unsqueeze(2)         # [B, Lq, L]
        attn_mask = (
            causal_mask
            & query_valid.unsqueeze(-1)
            & key_valid.unsqueeze(-2)
        )  # [B, Lq, L]
        return attn_mask.unsqueeze(1)  # [B, 1, Lq, L]

    def forward(self, s_tokens, ps_tokens, ns_tokens, kv_mask=None, q_mask=None):
        """
        s_tokens:  B x Ls x D
        ps_tokens: B x qLs x D
        ns_tokens: B x Lns x D
        kv_mask:   B x Ls,  valid mask for full S key/value
        q_mask:    B x qLs, valid mask for pruned S query
        """
        B, Ls, D = s_tokens.shape
        _, qLs, _ = ps_tokens.shape
        _, Lns, _ = ns_tokens.shape
        device = s_tokens.device

        if kv_mask is not None:
            s_tokens = s_tokens * kv_mask.unsqueeze(-1).float()
        if q_mask is not None:
            ps_tokens = ps_tokens * q_mask.unsqueeze(-1).float()

        # shared QKV for S-tokens
        q_s = self.W_q_s(ps_tokens)   # B x qLs x D
        k_s = self.W_k_s(s_tokens)    # B x Ls x D
        v_s = self.W_v_s(s_tokens)    # B x Ls x D

        # token-specific QKV for NS-tokens
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)

        q = torch.cat([q_s, q_ns], dim=1)   # B x (qLs+Lns) x D
        k = torch.cat([k_s, k_ns], dim=1)   # B x (Ls+Lns) x D
        v = torch.cat([v_s, v_ns], dim=1)   # B x (Ls+Lns) x D

        L = Ls + Lns
        Lq = qLs + Lns

        # split heads
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x Lq x h
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # B x H x L x h
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = self.build_mask(
            B=B,
            Ls=Ls,
            qLs=qLs,
            Lns=Lns,
            device=device,
            kv_mask=kv_mask,
            q_mask=q_mask
        )

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask
        )

        # concat heads
        output = output.transpose(1, 2).contiguous().view(B, Lq, D)
        output = self.W_o(output)

        ps_out = output[:, :qLs, :]
        ns_out = output[:, qLs:, :]

        if q_mask is not None:
            ps_out = ps_out * q_mask.unsqueeze(-1).float()

        return ps_out, ns_out


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
        # layer 1: B x Lns x D  @  Lns x D x H -> B x Lns x H
        ns_hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1_ns) + self.b1_ns.unsqueeze(0)
        ns_hidden = self.activation(ns_hidden)
        # layer 2: B x Lns x H  @  Lns x H x D -> B x Lns x D
        ns_out = torch.einsum("blh,lhd->bld", ns_hidden, self.W2_ns) + self.b2_ns.unsqueeze(0)

        if mask is not None:
            s_out = s_out * mask.unsqueeze(-1).float()

        return s_out, ns_out