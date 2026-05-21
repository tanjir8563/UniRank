import math
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, SwiGLU


class TokenFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="TokenFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_full_layers=2,
                 window_size=[32, 16],
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
        super(TokenFormer, self).__init__(feature_map,
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

        self.item_info_dim = 0
        self.non_item_dim = 0
        self.user_feature_num = 0

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
                self.user_feature_num += 1

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.unified_tokenizer_layer = Embedding2Tokenization(
            embedding_dim=embedding_dim,
            item_info_dim=self.item_info_dim,
            token_dim=token_dim,
            user_feature_num=self.user_feature_num,
        )

        self.unified_layers = UnifiedInteractionBlocks(
            token_dim=token_dim,
            num_full_layers=num_full_layers,
            window_size=window_size,
            user_feature_num=self.user_feature_num,
            expand=expansion_factor,
            num_head=num_head,
            net_dropout=net_dropout,
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
            assert len(task) == num_tasks, \
                "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([
                self.get_output_activation(str(t)) for t in task
            ])
        else:
            self.output_activation = nn.ModuleList([
                self.get_output_activation(task) for _ in range(num_tasks)
            ])

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :].unsqueeze(1)
        sequence_emb = item_feat_emb[:, 0:-1, :]

        user_profile = self.embedding_layer(batch_dict, flatten_emb=False)

        unified_tokens, unified_valid_mask = self.unified_tokenizer_layer(
            user_profile,
            sequence_emb,
            target_emb,
            seq_mask=mask,
        )

        unified_tokens = self.unified_layers(
            unified_tokens,
            valid_mask=unified_valid_mask,
        )

        tower_output = [
            self.tower[i](unified_tokens[:, -1, :])
            for i in range(self.num_tasks)
        ]

        y_pred = [
            self.output_activation[i](tower_output[i])
            for i in range(self.num_tasks)
        ]

        return_dict = {}
        labels = self.feature_map.labels

        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]

        return return_dict


class Embedding2Tokenization(nn.Module):
    def __init__(self, embedding_dim, item_info_dim, token_dim, user_feature_num):
        super(Embedding2Tokenization, self).__init__()

        self.token_dim = token_dim
        self.user_feature_num = user_feature_num

        self.W_u = nn.Parameter(
            torch.empty(user_feature_num, embedding_dim, token_dim)
        )
        nn.init.xavier_uniform_(self.W_u)

        self.W_s = nn.Sequential(
            nn.Linear(item_info_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )

        self.W_i = nn.Sequential(
            nn.Linear(item_info_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )

        self.SEP1 = nn.Parameter(torch.empty(1, 1, token_dim))
        self.SEP2 = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.xavier_uniform_(self.SEP1)
        nn.init.xavier_uniform_(self.SEP2)

    def forward(self, user_profile, sequence_emb, target_emb, seq_mask=None):
        """
        Args:
            user_profile: B x F x embedding_dim
            sequence_emb: B x T x item_info_dim
            target_emb:   B x 1 x item_info_dim
            seq_mask:     B x T, 1/True for valid seq positions

        Returns:
            unified_tokens: B x S x D
            unified_valid_mask: B x S, bool
        """
        batch_size, T, _ = sequence_emb.shape
        device = sequence_emb.device
        F = self.user_feature_num

        user_tokens = torch.einsum("bfd,fdk->bfk", user_profile, self.W_u)
        seq_tokens = self.W_s(sequence_emb)
        target_tokens = self.W_i(target_emb)

        SEP1 = self.SEP1.expand(batch_size, -1, -1)
        SEP2 = self.SEP2.expand(batch_size, -1, -1)

        unified_tokens = torch.cat(
            [user_tokens, SEP1, seq_tokens, SEP2, target_tokens],
            dim=1,
        )

        if seq_mask is None:
            seq_valid = torch.ones(batch_size, T, dtype=torch.bool, device=device)
        else:
            seq_valid = seq_mask.bool()

        valid_user = torch.ones(batch_size, F, dtype=torch.bool, device=device)
        valid_sep1 = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_sep2 = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_target = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        unified_valid_mask = torch.cat(
            [valid_user, valid_sep1, seq_valid, valid_sep2, valid_target],
            dim=1,
        )

        return unified_tokens, unified_valid_mask


class UnifiedInteractionBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_full_layers,
                 window_size=[32, 16],
                 user_feature_num=0,
                 expand=2,
                 num_head=2,
                 net_dropout=0.0):
        super(UnifiedInteractionBlocks, self).__init__()

        self.num_full_layers = num_full_layers
        self.window_size = window_size
        self.user_feature_num = user_feature_num
        self.num_layers = len(window_size) + num_full_layers

        if hasattr(nn, "RMSNorm"):
            self.mixer_norms = nn.ModuleList([
                nn.RMSNorm(token_dim)
                for _ in range(self.num_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.RMSNorm(token_dim)
                for _ in range(self.num_layers)
            ])
        else:
            self.mixer_norms = nn.ModuleList([
                nn.LayerNorm(token_dim)
                for _ in range(self.num_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(token_dim)
                for _ in range(self.num_layers)
            ])

        self.mixer_layers = nn.ModuleList([
            GatedMultiHeadAttentionLayer(
                token_dim=token_dim,
                num_head=num_head,
                user_feature_num=user_feature_num,
            )
            for _ in range(self.num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            SwiGLU(
                input_dim=token_dim,
                expand=expand,
                net_dropout=net_dropout,
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor):
        for i in range(self.num_layers):
            if i < self.num_full_layers:
                attn_mode = "full"
                cur_window_size = None
            else:
                attn_mode = "sliding"
                cur_window_size = self.window_size[i - self.num_full_layers]

            x = self.mixer_layers[i](
                self.mixer_norms[i](x),
                valid_mask=valid_mask,
                attn_mode=attn_mode,
                window_size=cur_window_size,
            ) + x

            x = self.ffn_layers[i](self.ffn_norms[i](x)) + x

            # 可选：每层结束后再次清理 padding token。
            # 这样可以避免 padding token 在残差分支中继续累积非零值。
            # 如果你只想严格做到“attention输出后乘 valid_mask”，可以删除这两行。
            if valid_mask is not None:
                x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class GatedMultiHeadAttentionLayer(nn.Module):
    def __init__(self, token_dim, num_head=2, user_feature_num=0):
        super(GatedMultiHeadAttentionLayer, self).__init__()

        assert token_dim % num_head == 0, "token_dim must be divisible by num_head"

        self.num_head = num_head
        self.head_dim = token_dim // num_head
        self.token_dim = token_dim
        self.user_feature_num = user_feature_num

        self.W_q = nn.Linear(token_dim, token_dim, bias=False)
        self.W_k = nn.Linear(token_dim, token_dim, bias=False)
        self.W_v = nn.Linear(token_dim, token_dim, bias=False)

        self.W_g = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=False),
            nn.Sigmoid(),
        )

        self._block_mask_cache = {}

    @staticmethod
    def _build_block_mask(S,
                          device,
                          num_head,
                          user_feature_num,
                          attn_mode="full",
                          window_size=None):
        """
        Unified stream:
            [user_tokens][SEP1][seq_tokens][SEP2][target]

        full:
            standard causal block mask

        sliding:
            sliding causal block mask
            + discard non-seq prefix [user_tokens]
        """
        if attn_mode == "full":
            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx <= q_idx
        else:
            assert window_size is not None

            prefix_len = int(user_feature_num)
            window = int(window_size)

            def mask_mod(b, h, q_idx, kv_idx):
                causal = kv_idx <= q_idx
                local = (q_idx - kv_idx) < window

                # 保留你原始语义：
                # sliding 层中，非 user token 不再看 user prefix。
                skip_prefix = (q_idx >= prefix_len) & (kv_idx < prefix_len)

                return causal & local & ~skip_prefix

        block_mask_kwargs = dict(
            mask_mod=mask_mod,
            B=None,
            H=num_head,
            Q_LEN=S,
            KV_LEN=S,
            device=device,
        )

        try:
            return create_block_mask(**block_mask_kwargs, _compile=False)
        except TypeError:
            return create_block_mask(**block_mask_kwargs)

    def _get_block_mask(self, S, device, attn_mode="full", window_size=None):
        device_key = (device.type, device.index)

        cache_key = (
            int(S),
            device_key,
            str(attn_mode),
            None if window_size is None else int(window_size),
            int(self.user_feature_num),
            int(self.num_head),
        )

        block_mask = self._block_mask_cache.get(cache_key)

        if block_mask is None:
            block_mask = self._build_block_mask(
                S=S,
                device=device,
                num_head=self.num_head,
                user_feature_num=self.user_feature_num,
                attn_mode=attn_mode,
                window_size=window_size,
            )
            self._block_mask_cache[cache_key] = block_mask

        return block_mask

    def forward(self,
                x: torch.Tensor,
                valid_mask=None,
                attn_mode="full",
                window_size=None):
        """
        Args:
            x: B x S x D
            valid_mask: B x S, bool. True means valid token.

        Important:
            This version does NOT introduce padding mask inside attention.
            It does NOT use score_mod to set invalid keys to -inf.
            It only multiplies the final attention output by valid_mask.
        """
        B, S, D = x.shape

        Q = self.W_q(x).view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_head, self.head_dim).transpose(1, 2)

        block_mask = self._get_block_mask(
            S=S,
            device=x.device,
            attn_mode=attn_mode,
            window_size=window_size,
        )

        attn_out = flex_attention(
            Q,
            K,
            V,
            block_mask=block_mask,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        attn_out = attn_out * self.W_g(x)

        if valid_mask is not None:
            attn_out = attn_out * valid_mask.unsqueeze(-1).to(dtype=attn_out.dtype)

        return attn_out