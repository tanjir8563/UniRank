import math
import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, PerTokenSwiGLU, DIN_Attention


class UniMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="UniMixer",
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
                 num_basis=4,
                 global_rank=4,
                 sinkhorn_iter=5,
                 tau=1.0,
                 tau_end=0.05,
                 tau_schedule="linear",
                 tau_anneal_steps=20000,
                 tau_warmup_steps=0,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(UniMixer, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps

        self.num_field = feature_map.get_num_fields()
        self.item_info_dim = 0
        self.item_feature_fields = 0

        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue

            emb_dim = spec.get("embedding_dim", embedding_dim)
            if emb_dim != embedding_dim:
                raise ValueError(
                    f"All feature embedding dims must equal embedding_dim={embedding_dim}, "
                    f"but feature `{feat}` has embedding_dim={emb_dim}."
                )

            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.item_feature_fields += 1
                self.num_field += 1

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )

        self.tokenizer_layer = Embedding2Tokenization(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_field=self.num_field,
            num_tasks=num_tasks
        )

        self.unified_layers = UniMixingLiteBlocks(
            token_dim=token_dim,
            num_token=self.num_field,
            num_layers=num_layers,
            expand=expansion_factor,
            num_basis=num_basis,
            global_rank=global_rank,
            sinkhorn_iter=sinkhorn_iter,
            tau_start=tau,
            tau_end=tau_end,
            tau_schedule=tau_schedule,
            tau_anneal_steps=tau_anneal_steps,
            tau_warmup_steps=tau_warmup_steps,
            net_dropout=net_dropout
        )

        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=token_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
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

    def get_current_tau(self):
        return self.unified_layers.get_current_tau()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, :-1, :]

        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)

        batch_emb = self.embedding_layer(batch_dict, flatten_emb=False)

        target_tokens = target_emb.view(batch_size, self.item_feature_fields, self.embedding_dim)
        pooling_tokens = pooling_emb.view(batch_size, self.item_feature_fields, self.embedding_dim)

        tokens = torch.cat([batch_emb, target_tokens, pooling_tokens], dim=1)

        tokens = self.tokenizer_layer(tokens)

        # 使用 BaseModel.fit() 里维护的 _total_steps 做温度调度
        self.unified_layers.update_tau(self._total_steps)
        tokens = self.unified_layers(tokens)

        bottom_output = tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict[f"{labels[i]}_pred"] = y_pred[i]
        return return_dict


class Embedding2Tokenization(nn.Module):
    def __init__(self, embedding_dim, token_dim, num_field, num_tasks):
        super(Embedding2Tokenization, self).__init__()
        self.token_dim = token_dim
        self.num_field = num_field
        self.num_tasks = num_tasks
        self.W = nn.Parameter(torch.empty(num_field, embedding_dim, token_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, feature_embeddings):
        tokens = torch.einsum("bfd,fdk->bfk", feature_embeddings, self.W)
        return tokens


class UniMixingLiteBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_layers,
                 expand=2,
                 num_basis=4,
                 global_rank=4,
                 sinkhorn_iter=5,
                 tau_start=1.0,
                 tau_end=None,
                 tau_schedule="cosine",
                 tau_anneal_steps=None,
                 tau_warmup_steps=0,
                 net_dropout=0.0):
        super(UniMixingLiteBlocks, self).__init__()

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_start if tau_end is None else tau_end)
        self.tau_schedule = str(tau_schedule).lower()
        self.tau_anneal_steps = None if tau_anneal_steps is None else int(tau_anneal_steps)
        self.tau_warmup_steps = int(tau_warmup_steps)

        if self.tau_start <= 0 or self.tau_end <= 0:
            raise ValueError("tau_start and tau_end must be positive.")

        if self.tau_schedule not in ["cosine", "linear", "exp"]:
            raise ValueError("tau_schedule must be one of ['cosine', 'linear', 'exp'].")

        self.register_buffer("current_tau", torch.tensor(self.tau_start, dtype=torch.float32))

        self.mixers = nn.ModuleList([
            UniMixingLiteLayer(
                token_dim=token_dim,
                num_token=num_token,
                num_basis=num_basis,
                global_rank=global_rank,
                sinkhorn_iter=sinkhorn_iter,
                tau=tau_start,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

        self.pffns = nn.ModuleList([
            SiameseFeedForwardLayer(
                token_dim=token_dim,
                num_token=num_token,
                expand=expand,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = build_norm(token_dim)

        # 初始化一次温度，确保 layer 内 current_tau 同步
        self.update_tau(step=0)

    def _compute_tau(self, step: int):
        step = max(0, int(step))

        if self.tau_anneal_steps is None or self.tau_anneal_steps <= 0:
            return self.tau_start

        if step <= self.tau_warmup_steps:
            return self.tau_start

        progress = (step - self.tau_warmup_steps) / max(1, self.tau_anneal_steps)
        progress = min(max(progress, 0.0), 1.0)

        if self.tau_schedule == "linear":
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
        elif self.tau_schedule == "cosine":
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1.0 + math.cos(math.pi * progress))
        elif self.tau_schedule == "exp":
            tau = self.tau_start * ((self.tau_end / self.tau_start) ** progress)
        else:
            raise ValueError(f"Unsupported tau_schedule={self.tau_schedule}")

        return max(float(tau), 1e-6)

    def update_tau(self, step: int):
        tau = self._compute_tau(step)
        self.current_tau.fill_(tau)
        for mixer in self.mixers:
            mixer.set_tau(tau)
        return tau

    def get_current_tau(self):
        return float(self.current_tau.item())

    def forward(self, x: torch.Tensor):
        x_stream = x
        y_stream = x

        for mixer, pffn in zip(self.mixers, self.pffns):
            x_stream, y_stream = mixer(x_stream, y_stream)
            x_stream, y_stream = pffn(x_stream, y_stream)

        out = x_stream + self.final_norm(y_stream)
        return out


class UniMixingLiteLayer(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_basis=4,
                 global_rank=4,
                 sinkhorn_iter=5,
                 tau=1.0,
                 net_dropout=0.0):
        super(UniMixingLiteLayer, self).__init__()

        self.token_dim = token_dim
        self.num_token = num_token
        self.num_basis = num_basis
        self.global_rank = global_rank
        self.sinkhorn_iter = sinkhorn_iter

        self.register_buffer("current_tau", torch.tensor(float(tau), dtype=torch.float32))

        # local mixing: token-specific coefficients over shared basis
        self.local_basis = nn.Parameter(
            torch.empty(num_basis, token_dim, token_dim)
        )
        self.local_coef = nn.Parameter(
            torch.empty(num_token, num_basis)
        )

        # global mixing: low-rank token mixing
        self.global_left = nn.Parameter(
            torch.empty(num_token, global_rank)
        )
        self.global_right = nn.Parameter(
            torch.empty(global_rank, num_token)
        )

        # SiameseNorm 需要的两个 norm
        self.norm_x = build_norm(token_dim)  # bounded stream
        self.norm_y = build_norm(token_dim)  # unbounded stream before fusion

        self.out_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.dropout = nn.Dropout(net_dropout)

        self.reset_parameters()

    def set_tau(self, tau: float):
        self.current_tau.fill_(max(float(tau), 1e-6))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.local_basis)
        nn.init.xavier_uniform_(self.local_coef)
        nn.init.xavier_uniform_(self.global_left)
        nn.init.xavier_uniform_(self.global_right)
        nn.init.xavier_uniform_(self.out_proj.weight)

    @staticmethod
    def sinkhorn(mat: torch.Tensor, n_iter: int = 5, eps: float = 1e-6):
        for _ in range(n_iter):
            mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
            mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
        return mat

    def apply_sinkhorn_constraint(self, logits: torch.Tensor):
        """
        logits: [..., N, N]
        先做温度缩放，再指数映射，再做 Sinkhorn 归一化
        """
        tau = self.current_tau.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
        logits = logits / tau
        logits = logits - torch.amax(logits, dim=(-1, -2), keepdim=True)
        mat = torch.exp(logits)
        mat = self.sinkhorn(mat, n_iter=self.sinkhorn_iter)
        return mat

    def local_mixing(self, x: torch.Tensor):
        """
        x: [B, T, D]
        token-specific mixing using shared basis + symmetry + Sinkhorn
        """
        local_coef = F.softmax(self.local_coef, dim=-1)  # [T, R]
        W_local_logits = torch.einsum("tr,rij->tij", local_coef, self.local_basis)  # [T, D, D]

        # 对称化，贴近论文中的对称约束
        W_local_logits = 0.5 * (W_local_logits + W_local_logits.transpose(-1, -2))

        # Sinkhorn 约束，贴近论文中的双随机约束
        W_local = self.apply_sinkhorn_constraint(W_local_logits)  # [T, D, D]

        x_local = torch.einsum("bti,tij->btj", x, W_local)  # [B, T, D]
        return x_local

    def build_global_mixing_logits(self):
        """
        low-rank token mixing logits
        return: [T, T]
        """
        G_logits = torch.matmul(self.global_left, self.global_right)
        G_logits = G_logits / math.sqrt(self.global_rank)
        return G_logits

    def symmetrize_global_mixing(self, G_logits: torch.Tensor):
        """
        对称化 global mixing
        """
        return 0.5 * (G_logits + G_logits.transpose(-1, -2))

    def global_mixing(self, G: torch.Tensor, x_local: torch.Tensor):
        """
        G: [T, T]
        x_local: [B, T, D]
        """
        return torch.einsum("ts,bsd->btd", G, x_local)

    def mixing_transform(self, x: torch.Tensor):
        """
        F_i: current UniMixing-Lite transform
        """
        x_local = self.local_mixing(x)

        G_logits = self.build_global_mixing_logits()
        G_logits = self.symmetrize_global_mixing(G_logits)
        G = self.apply_sinkhorn_constraint(G_logits)

        y = self.global_mixing(G, x_local)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y

    def siamese_norm(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        """
        按论文更接近的双流拓扑：

        Y'_i    = LN_Y(Y_i)
        O_i     = F_i(X_i + Y'_i)
        X_{i+1} = LN_X(X_i + O_i)
        Y_{i+1} = Y_i + O_i
        """
        y_norm = self.norm_y(y_stream)
        fused_input = x_stream + y_norm

        update = self.mixing_transform(fused_input)

        x_next = self.norm_x(x_stream + update)
        y_next = y_stream + update
        return x_next, y_next

    def forward(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        return self.siamese_norm(x_stream, y_stream)


class SiameseFeedForwardLayer(nn.Module):
    """
    对 PFFN 使用同样的 SiameseNorm 双流更新
    """
    def __init__(self,
                 token_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0):
        super(SiameseFeedForwardLayer, self).__init__()

        self.norm_x = build_norm(token_dim)
        self.norm_y = build_norm(token_dim)

        self.pffn = PerTokenSwiGLU(
            input_dim=token_dim,
            num_token=num_token,
            expand=expand,
            net_dropout=net_dropout
        )

    def siamese_norm(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        y_norm = self.norm_y(y_stream)
        fused_input = x_stream + y_norm

        update = self.pffn(fused_input)

        x_next = self.norm_x(x_stream + update)
        y_next = y_stream + update
        return x_next, y_next

    def forward(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        return self.siamese_norm(x_stream, y_stream)


def build_norm(normalized_shape):
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(normalized_shape)
    return nn.LayerNorm(normalized_shape)