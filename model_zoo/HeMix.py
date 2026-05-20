import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, PerTokenFeedForward


class HeMix(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="HeMix",
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
                 num_glocal_token=4,
                 num_real_token=4,
                 num_mix_heads=4,
                 low_rank_dim=None,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 real_seq_len=20,
                 **kwargs):
        super(HeMix, self).__init__(feature_map,
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
        self.num_glocal_token = num_glocal_token
        self.num_real_token = num_real_token
        self.accumulation_steps = accumulation_steps
        self.real_seq_len = real_seq_len

        # 统计 item 特征维度 / 非 item 特征维度
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

        # 论文中：先生成 NS tokens，再基于其与 fixed queries 对 sequence 做 mixed attention
        tokenizer_input_dim = self.non_item_dim + self.item_info_dim
        self.tokenizer_layer = Embedding2Tokenization(
            input_dim=tokenizer_input_dim,
            token_dim=token_dim,
            num_glocal_token=num_glocal_token,
            num_real_token=num_real_token
        )

        # 序列 item / target item 投影到统一 token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # 总 token 数 = sequence tokens + NS tokens
        # NS token 数 = num_glocal_token + num_real_token
        # Seq token 数 = 2 * NS token 数（global / real 中都包含 dynamic + fixed queries）
        self.num_ns_token = num_glocal_token + num_real_token
        self.num_all_tokens = self.num_ns_token * 3

        self.unified_layers = HeteroMixer(
            input_dim=token_dim,
            num_token=self.num_all_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            num_mix_heads=num_mix_heads,
            low_rank_dim=low_rank_dim,
            net_dropout=net_dropout
        )

        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim,
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout,
                      batch_norm=batch_norm)
            for _ in range(num_tasks)
        ])

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

        if mask is not None:
            batch_size = mask.shape[0]
        else:
            any_tensor = next(iter(item_dict.values()))
            batch_size = any_tensor.shape[0]

        # item_dict: [history_items..., target_item]
        # flatten_emb=True 后 reshape 成 B x (T+1) x item_info_dim
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_feat_emb[:, :-1, :]   # B x S x item_info_dim

        # real-time sequence: 最近 real_seq_len 个行为
        # global sequence: 更早历史行为
        real_sequence_emb = sequence_emb[:, -self.real_seq_len:, :]
        glocal_sequence_emb = sequence_emb[:, :-self.real_seq_len, :]

        # 投影到 token_dim，供 mixed hetero attention 使用
        real_sequence_tokens = self.item_token_proj(real_sequence_emb)
        glocal_sequence_tokens = self.item_token_proj(glocal_sequence_emb)

        real_mask, glocal_mask = None, None
        if mask is not None:
            seq_mask = mask[:, :-1].bool()  # 去掉 target item 对应位置
            real_mask = seq_mask[:, -self.real_seq_len:]
            glocal_mask = seq_mask[:, :-self.real_seq_len]

        # 非序列特征 + target item -> NS tokens
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # B x non_item_dim
        feature_emb = torch.cat([batch_emb, target_emb], dim=-1)

        # tokens: [sequence_tokens, ns_tokens]
        tokens = self.tokenizer_layer(
            feature_emb,
            real_sequence_tokens,
            glocal_sequence_tokens,
            real_mask=real_mask,
            glocal_mask=glocal_mask
        )

        # HeteroMixer interaction
        tokens = self.unified_layers(tokens)

        # prediction
        bottom_output = tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class Embedding2Tokenization(nn.Module):
    """
    对应论文 3.3:
    1) 非序列特征 -> NS tokens
    2) NS tokens 分为 global / real 两部分
    3) 各自与 fixed query 拼接
    4) 分别对 global / real sequence 做 Mixed Hetero Attention
    5) 最终输出 [sequence_tokens, ns_tokens]
    """
    def __init__(self, input_dim, token_dim, num_glocal_token, num_real_token, num_heads=2):
        super(Embedding2Tokenization, self).__init__()
        self.num_glocal_token = num_glocal_token
        self.num_real_token = num_real_token
        self.num_total_ns_tokens = num_glocal_token + num_real_token
        self.token_dim = token_dim

        output_dim = token_dim * self.num_total_ns_tokens
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

        # fixed queries
        self.cls_tokens_glocal = nn.Parameter(torch.empty(1, num_glocal_token, token_dim))
        self.cls_tokens_real = nn.Parameter(torch.empty(1, num_real_token, token_dim))

        self.mha_glocal = MixedHeteroAttention(
            input_dim=token_dim,
            num_heads=num_heads,
            num_ns_token=num_glocal_token * 2
        )
        self.mha_real = MixedHeteroAttention(
            input_dim=token_dim,
            num_heads=num_heads,
            num_ns_token=num_real_token * 2
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cls_tokens_glocal)
        nn.init.xavier_uniform_(self.cls_tokens_real)

    def forward(self, feature_embeddings, real_sequence_emb, glocal_sequence_emb,
                real_mask=None, glocal_mask=None):
        """
        feature_embeddings: B x input_dim
        real_sequence_emb:  B x Lr x D
        glocal_sequence_emb: B x Lg x D
        return: B x (3 * num_total_ns_tokens) x token_dim
        """
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)

        B = feature_embeddings.size(0)

        # 1) non-sequential features -> NS tokens
        flatten_ns_tokens = self.mlp(feature_embeddings)
        ns_tokens = flatten_ns_tokens.view(B, self.num_total_ns_tokens, self.token_dim)

        ns_tokens_glocal = ns_tokens[:, :self.num_glocal_token, :]
        ns_tokens_real = ns_tokens[:, self.num_glocal_token:, :]

        # 2) mixed queries = dynamic queries (NS tokens) + fixed queries
        fixed_real = self.cls_tokens_real.expand(B, -1, -1)
        fixed_glocal = self.cls_tokens_glocal.expand(B, -1, -1)

        real_queries = torch.cat([ns_tokens_real, fixed_real], dim=1)
        glocal_queries = torch.cat([ns_tokens_glocal, fixed_glocal], dim=1)

        # 3) mixed hetero attention over real/global sequences
        tokens_real = self.mha_real(
            s_tokens=real_sequence_emb,
            ns_tokens=real_queries,
            mask=real_mask
        )
        tokens_glocal = self.mha_glocal(
            s_tokens=glocal_sequence_emb,
            ns_tokens=glocal_queries,
            mask=glocal_mask
        )

        # 4) final tokens = [sequence_tokens, ns_tokens]
        seq_tokens = torch.cat([tokens_real, tokens_glocal], dim=1)
        tokens = torch.cat([seq_tokens, ns_tokens], dim=1)
        return tokens


class HeteroMixer(nn.Module):
    """
    对应论文 3.4:
    [HeteroMixing + AddNorm] -> [HeteroFFN + AddNorm]
    """
    def __init__(self,
                 input_dim,
                 num_token,
                 num_layers,
                 expand=2,
                 num_mix_heads=4,
                 low_rank_dim=None,
                 net_dropout=0.0):
        super(HeteroMixer, self).__init__()
        self.num_layers = num_layers

        if hasattr(nn, "RMSNorm"):
            self.mixer_norms = nn.ModuleList([nn.RMSNorm(input_dim) for _ in range(num_layers)])
            self.pffn_norms = nn.ModuleList([nn.RMSNorm(input_dim) for _ in range(num_layers)])
        else:
            self.mixer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
            self.pffn_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

        self.mixer_layers = nn.ModuleList([
            MultiHeadTokenMixing(
                input_dim=input_dim,
                num_token=num_token,
                num_heads=num_mix_heads,
                low_rank_dim=low_rank_dim
            )
            for _ in range(num_layers)
        ])

        # 沿用原有 PerTokenFeedForward，最小侵入式实现 HeteroFFN
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(
                input_dim=input_dim,
                num_token=num_token,
                expand=expand,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x


class MixedHeteroAttention(nn.Module):
    """
    对应论文 3.3.2:
    - query 侧使用 token-specific heterogeneous projection
    - key/value 侧共享投影
    """
    def __init__(self, input_dim, num_heads, num_ns_token):
        super(MixedHeteroAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads

        # heterogeneous projection for queries
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))

        # shared projection for sequence keys / values
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_s.weight)
        nn.init.xavier_uniform_(self.W_v_s.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x Ls x D
        ns_tokens: B x Lns x D
        mask:      B x Ls, True/1 for valid positions
        """
        B, Ls, D = s_tokens.shape
        _, Lns, _ = ns_tokens.shape

        # 若该段序列为空，则直接保留 queries
        if Ls == 0:
            return ns_tokens

        # token-specific query projection
        q_weight = self.W_q_ns[:Lns]  # Lns x D x D
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, q_weight)  # B x Lns x D

        # shared key/value projection
        k_s = self.W_k_s(s_tokens)  # B x Ls x D
        v_s = self.W_v_s(s_tokens)  # B x Ls x D

        # split heads
        q = q_ns.view(B, Lns, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x Lns x Dh
        k = k_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)    # B x H x Ls x Dh
        v = v_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)    # B x H x Ls x Dh

        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # concat heads
        output = output.transpose(1, 2).contiguous().view(B, Lns, D)
        output = self.W_o(output)
        return output


class MultiHeadTokenMixing(nn.Module):
    """
    对应论文 3.4.1 的 HeteroMixing:
    1) Multi-head Token Fusion
    2) Head-wise low-rank interaction
    3) Reconstruction
    """
    def __init__(self, input_dim, num_token, num_heads=4, low_rank_dim=None):
        super(MultiHeadTokenMixing, self).__init__()
        assert input_dim % num_heads == 0, \
            "input_dim must be divisible by num_heads"

        self.num_token = num_token
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.mix_dim = num_token * self.head_dim
        if low_rank_dim is None:
            low_rank_dim = max(16, self.mix_dim // 8)
        self.low_rank_dim = low_rank_dim
        # 每个 head 一个低秩 MLP: W_l / W_r
        self.W_l = nn.Parameter(torch.empty(num_heads, self.mix_dim, low_rank_dim))
        self.b_l = nn.Parameter(torch.zeros(num_heads, low_rank_dim))
        self.W_r = nn.Parameter(torch.empty(num_heads, low_rank_dim, self.mix_dim))
        self.b_r = nn.Parameter(torch.zeros(num_heads, self.mix_dim))
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.zeros_(self.b_l)
        nn.init.zeros_(self.b_r)

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        B, T, D = x.shape
        assert T == self.num_token, \
            "token number mismatch: expected {}, got {}".format(self.num_token, T)

        # 1) Multi-head Token Fusion
        # [B, T, D] -> [B, H, T, Dh] -> [B, H, T*Dh]
        mixed = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mixed = mixed.reshape(B, self.num_heads, self.mix_dim)

        # 2) Heterogeneous Mixed-token Interaction (head-wise low-rank MLP)
        hidden = torch.einsum("bhf,hfr->bhr", mixed, self.W_l) + self.b_l.unsqueeze(0)
        hidden = self.act(hidden)
        out = torch.einsum("bhr,hrf->bhf", hidden, self.W_r) + self.b_r.unsqueeze(0)

        # 3) Reconstruction
        out = out.view(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, T, D)
        return out