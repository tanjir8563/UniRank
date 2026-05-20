import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class UltraHSTU(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="UltraHSTU",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 truncation_start_layer=1,
                 truncation_ratio=0.5,
                 k1=[32, 32, 16],
                 k2=[20, 10, 10],
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_head=2,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(UltraHSTU, self).__init__(feature_map,
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
        self.truncation_start_layer = truncation_start_layer
        self.truncation_ratio = truncation_ratio

        assert len(k1) == len(k2) and truncation_start_layer <= len(k1)
        num_layers = len(k1)

        # 统计非 item 特征维度、item 特征维度
        self.item_info_dim = 0
        self.user_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item"]:
                self.item_info_dim += emb_dim
            elif spec.get("source") not in ["item", "action"]:
                self.user_info_dim += emb_dim

        # action 单独 embedding，其余 feature 走常规 embedding
        self.action_embedding_layer = FeatureEmbedding(
            feature_map, self.item_info_dim, required_feature_columns=['action']
        )
        self.feature_embedding_layer = FeatureEmbedding(
            feature_map, embedding_dim, not_required_feature_columns=['action']
        )

        self.unified_tokenizer_layer = Embedding2Tokenization(
            user_info_dim=self.user_info_dim,
            item_info_dim=self.item_info_dim,
            token_dim=token_dim
        )
        self.unified_layers = UnifiedInteractionBlocks(
            token_dim=token_dim,
            num_layers=num_layers,
            truncation_start_layer=truncation_start_layer,
            truncation_ratio=truncation_ratio,
            k1=k1,
            k2=k2,
            num_head=num_head,
            expansion_factor=expansion_factor,
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
            assert len(task) == num_tasks
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(str(t)) for t in task]
            )
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

        item_feat_emb = self.feature_embedding_layer(item_dict, flatten_emb=True)
        action_emb = self.action_embedding_layer(item_dict, flatten_emb=True)
        sequence_emb = item_feat_emb + action_emb                              # B x (T+1) x item_info_dim

        sequence_emb = sequence_emb.view(batch_size, -1, self.item_info_dim)   # B x (T+1) x item_info_dim

        # 用户上下文
        user_profile = self.feature_embedding_layer(batch_dict, flatten_emb=True) # B x user_info_dim

        # [user_token][sequence_tokens|candidate]
        unified_tokens = self.unified_tokenizer_layer(user_profile, sequence_emb) # B x (1 + T+1) x D

        unified_tokens = self.unified_layers(unified_tokens, valid_mask=mask)

        final_token = unified_tokens[:, -1, :]
        tower_output = [self.tower[i](final_token) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict[f"{labels[i]}_pred"] = y_pred[i]
        return return_dict


class Embedding2Tokenization(nn.Module):
    """
    将 user context 压成 1 个前缀 token，
    序列部分保留为逐位置 token。
    """
    def __init__(self, user_info_dim, item_info_dim, token_dim):
        super(Embedding2Tokenization, self).__init__()
        self.token_dim = token_dim
        self.mlp_u = nn.Sequential(
            nn.Linear(user_info_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim)
        )
        self.mlp_i = nn.Sequential(
            nn.Linear(item_info_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim)
        )

    def forward(self, user_profile, sequence_emb):
        """
        user_profile: B x user_info_dim
        sequence_emb: B x L x item_info_dim

        return:
            unified_tokens: B x (1+L) x D
        """
        user_token = self.mlp_u(user_profile).unsqueeze(1)  # B x 1 x D
        seq_tokens = self.mlp_i(sequence_emb)               # B x L x D
        return torch.cat([user_token, seq_tokens], dim=1)


class UnifiedInteractionBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_layers,
                 truncation_start_layer,
                 truncation_ratio,
                 k1,
                 k2,
                 num_head=2,
                 expansion_factor=4,
                 net_dropout=0.0):
        super(UnifiedInteractionBlocks, self).__init__()
        self.num_layers = num_layers
        self.truncation_start_layer = truncation_start_layer
        self.truncation_ratio = truncation_ratio
        self.k1 = k1
        self.k2 = k2

        self.stu_layers = nn.ModuleList([
            SequentialTransductionUnit(
                token_dim=token_dim,
                num_head=num_head,
                expansion_factor=expansion_factor,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

    def _expand_valid_mask(self, x, valid_mask):
        """
        输入 valid_mask 只覆盖历史 T。
        sequence_emb 实际长度是 T+1（最后一个是 candidate）。
        unified token 还需要额外补 1 个 user token。
        """
        if valid_mask is None:
            return None

        B = x.size(0)
        device = x.device
        valid_mask = valid_mask.bool()

        user_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        candidate_valid = torch.ones(B, 1, dtype=torch.bool, device=device)

        # [user][history][candidate]
        return torch.cat([user_valid, valid_mask, candidate_valid], dim=1)

    def truncate_tail(self, x, valid_mask):
        """
        保留:
        - 1 个 user token
        - 最近 truncation_ratio 比例的 sequence tokens
        """
        if self.truncation_ratio is None:
            return x, valid_mask

        B, S, D = x.shape
        seq_len = S - 1
        if seq_len <= 1:
            return x, valid_mask

        keep_seq_len = max(1, int(seq_len * self.truncation_ratio))
        keep_seq_len = min(keep_seq_len, seq_len)

        x = torch.cat([x[:, :1, :], x[:, -keep_seq_len:, :]], dim=1)

        if valid_mask is not None:
            valid_mask = torch.cat([valid_mask[:, :1], valid_mask[:, -keep_seq_len:]], dim=1)

        return x, valid_mask

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor):
        unified_valid_mask = self._expand_valid_mask(x, valid_mask)

        for i in range(self.num_layers):
            if i == self.truncation_start_layer:
                x, unified_valid_mask = self.truncate_tail(x, unified_valid_mask)

            x = self.stu_layers[i](
                x,
                valid_mask=unified_valid_mask,
                k1=self.k1[i],
                k2=self.k2[i]
            ) + x

        return x


class SequentialTransductionUnit(nn.Module):
    def __init__(self, token_dim, num_head=2, expansion_factor=4, net_dropout=0.0):
        super(SequentialTransductionUnit, self).__init__()
        assert token_dim % num_head == 0
        self.num_head = num_head
        self.head_dim = token_dim // num_head
        self.token_dim = token_dim

        self.pre_norm = nn.RMSNorm(token_dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(token_dim)
        self.attn_out_norm = nn.RMSNorm(token_dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(token_dim)

        hidden_dim = token_dim * expansion_factor

        self.pre_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(net_dropout),
            nn.Linear(hidden_dim, token_dim * 4, bias=False)
        )

        self.post_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(net_dropout),
            nn.Linear(hidden_dim, token_dim)
        )

    @staticmethod
    def _build_sparse_block_mask(B, H, S, device, k1, k2):
        """
        token 结构: [user][history][candidate]
        约束:
        - user token query 只看自己
        - sequence query 总是允许看 user token
        - sequence 内部使用 sparse:
            1) 局部 K1 滑窗（因果）
            2) 最近 K2 query 看全历史（因果）
        """

        seq_len = S - 1

        def mask_mod(b, h, q_idx, kv_idx):
            is_user_query = (q_idx == 0)
            is_user_key = (kv_idx == 0)

            # user query 只能看自己
            user_rule = (kv_idx == 0)

            # sequence 子空间索引
            q_seq_idx = q_idx - 1
            kv_seq_idx = kv_idx - 1

            is_causal = kv_seq_idx <= q_seq_idx
            in_local_window = (q_seq_idx - kv_seq_idx) < k1
            in_recent_queries = q_seq_idx >= (seq_len - k2)

            seq_rule = is_causal & (in_local_window | in_recent_queries)

            return torch.where(
                is_user_query,
                user_rule,
                is_user_key | seq_rule
            )

        return create_block_mask(mask_mod, B, H, S, S, device=device)

    def forward(self, x: torch.Tensor, valid_mask=None, k1=32, k2=20):
        B, S, D = x.shape

        norm_x = self.pre_norm(x)

        U, Q, K, V = self.pre_proj(norm_x).chunk(4, dim=-1)

        Q = Q.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_head, self.head_dim).transpose(1, 2)

        sparse_block_mask = self._build_sparse_block_mask(
            B=B,
            H=self.num_head,
            S=S,
            device=x.device,
            k1=k1,
            k2=k2
        )

        A = flex_attention(
            Q,
            K,
            V,
            block_mask=sparse_block_mask,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        A = A.transpose(1, 2).contiguous().view(B, S, D)

        Y = self.post_proj(self.attn_out_norm(A) * F.silu(U))

        if valid_mask is not None:
            Y = Y * valid_mask.detach().unsqueeze(-1).to(dtype=Y.dtype)
        return Y