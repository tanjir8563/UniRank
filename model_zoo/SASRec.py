# =========================================================================
# SASRec next-item pretraining (UniRank dataloader 适配)
# item_dict[item_id_key]: [B, T+1] = 历史 T 步 + target；mask: [B, T]
#
# 训练（default）：
#   将本 batch 内所有「非 pad 且 label>0」的位置展平为 M 条，每条用序列表征 Q 与
#   正样本 item 向量、批内易负、批内难负（其他序列的正样本向量）拼 logits，
#   cross_entropy(..., target=0) → 等价于在 2M+1 类上做 softmax，正类为第 0 列。
#   每个监督位置的分类数 = 2M + 1（M 随 batch 变化，不是固定超参）。
#
# 可选 loss_type=sampled_ce：正类 + K 随机负的 softmax（省显存的全词表近似）。
# =========================================================================

import sys
import logging

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from fuxictr.pytorch.models import BaseModel
from fuxictr.utils import not_in_whitelist


class SASRecBlock(nn.Module):
    """带因果掩码的 Self-Attention + FFN（与常见 SASRec 一致）。"""

    def __init__(self, hidden_size, num_heads, dropout_rate=0.0):
        super(SASRecBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask, attn_mask):
        attn_out, _ = self.attention(
            x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask
        )
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class SASRecPretrain(BaseModel):
    """
    纯序列 next-item：item_id embedding + 因果 Transformer。
    loss_type:
      - infonce_batch: 与 TAAC 相同的 batch 内 1+易负(M)+难负(M) 对比 softmax
      - sampled_ce: 每位置 1+K 类 softmax（K=num_sampled_negs）
    """

    def __init__(
        self,
        feature_map,
        model_id="SASRecPretrain",
        gpu=-1,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        max_seq_len=100,
        dropout_rate=0.1,
        item_id_key="item_id",
        learning_rate=1e-3,
        accumulation_steps=1,
        loss_type="infonce_batch",
        temperature=0.07,
        num_sampled_negs=128,
        eval_vocab_chunk=65536,
        infonce_mask_value=-1e9,
        infonce_max_supervised=4096,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs,
    ):
        task = kwargs.pop("task", "binary_classification")
        # YAML 命名与 TAAC（hidden_units / num_blocks / maxlen）及其它实验（embedding_dim / max_len / net_dropout）对齐
        if "hidden_size" in kwargs:
            hidden_size = int(kwargs.pop("hidden_size"))
        elif "hidden_units" in kwargs:
            hidden_size = int(kwargs.pop("hidden_units"))
        elif "embedding_dim" in kwargs:
            hidden_size = int(kwargs.pop("embedding_dim"))
        if "max_len" in kwargs:
            max_seq_len = int(kwargs.pop("max_len"))
        if "net_dropout" in kwargs:
            dropout_rate = float(kwargs.pop("net_dropout"))
        if "num_blocks" in kwargs:
            num_layers = int(kwargs.pop("num_blocks"))
        weight_decay = float(kwargs.pop("weight_decay", 0) or 0)

        super(SASRecPretrain, self).__init__(
            feature_map=feature_map,
            model_id=model_id,
            task=task,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        if item_id_key not in feature_map.features:
            raise KeyError(
                "item_id_key={} not in feature_map; set item_id_key to your item id column name.".format(
                    item_id_key
                )
            )
        self.item_id_key = item_id_key
        self.item_vocab_size = int(feature_map.features[item_id_key]["vocab_size"])
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.accumulation_steps = accumulation_steps
        self.loss_type = str(loss_type).lower()
        self.temperature = float(temperature)
        self.num_sampled_negs = int(num_sampled_negs)
        self.eval_vocab_chunk = int(eval_vocab_chunk)
        self.infonce_mask_value = float(infonce_mask_value)
        self.infonce_max_supervised = int(infonce_max_supervised)

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.item_embedding = nn.Embedding(self.item_vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.sasrec_blocks = nn.ModuleList(
            [SASRecBlock(hidden_size, num_heads, dropout_rate) for _ in range(num_layers)]
        )

        self.compile(kwargs["optimizer"], "binary_cross_entropy", learning_rate)
        if weight_decay > 0:
            opt = str(kwargs.get("optimizer", "adam")).lower()
            if opt == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif opt == "sgd":
                self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.reset_parameters()
        self.model_to_device()

    def get_inputs(self, inputs, feature_source=None):
        if len(inputs) == 4:
            batch_dict, item_dict, mask, multi_masks = inputs
        elif len(inputs) == 3:
            batch_dict, item_dict, mask = inputs
            multi_masks = None
        else:
            raise ValueError(f"Unexpected inputs length: {len(inputs)}. Expected 3 or 4.")

        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def _encode_sequence(self, item_seq):
        batch_size, seq_len = item_seq.size()
        seq_emb = self.item_embedding(item_seq)
        positions = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len).clamp(max=self.max_seq_len - 1)
        pos_emb = self.position_embedding(positions)
        seq_emb = self.emb_dropout(seq_emb + pos_emb)
        seq_emb = seq_emb.masked_fill((item_seq == 0).unsqueeze(-1), 0.0)

        padding_mask = item_seq == 0
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device, dtype=torch.float32) * float("-inf"),
            diagonal=1,
        )

        for block in self.sasrec_blocks:
            seq_emb = block(seq_emb, padding_mask, causal_mask)
        return seq_emb

    def _scores_vs_item_weights(self, seq_repr, chunk=None):
        if chunk is None:
            chunk = self.eval_vocab_chunk
        W = self.item_embedding.weight
        parts = []
        for start in range(0, self.item_vocab_size, chunk):
            end = min(start + chunk, self.item_vocab_size)
            parts.append(torch.mm(seq_repr, W[start:end].T))
        return torch.cat(parts, dim=1)

    def _argmax_predict_from_last_hidden(self, seq_repr):
        W = self.item_embedding.weight
        chunk = self.eval_vocab_chunk
        Bv = seq_repr.size(0)
        best_score = torch.full((Bv,), float("-inf"), device=seq_repr.device, dtype=seq_repr.dtype)
        best_idx = torch.zeros(Bv, dtype=torch.long, device=seq_repr.device)
        for start in range(0, self.item_vocab_size, chunk):
            end = min(start + chunk, self.item_vocab_size)
            scores = torch.mm(seq_repr, W[start:end].T)
            local_max, local_arg = scores.max(dim=1)
            better = local_max > best_score
            best_score = torch.where(better, local_max, best_score)
            best_idx = torch.where(better, start + local_arg, best_idx)
        return best_idx

    def _infonce_batch_loss(self, seq_emb, item_seq, labels):
        """
        TAAC 风格：展平 M 个监督位置，每行 logits 维度 1 + M + M，softmax 目标类 0。
        每个位置的「分类数」= 2M + 1（M 为本 batch 监督条数）。
        """
        sup = (item_seq != 0) & (labels > 0)
        if not sup.any():
            return seq_emb.sum() * 0.0

        b_ix, t_ix = sup.nonzero(as_tuple=True)
        Q = seq_emb[b_ix, t_ix, :]
        pos_ids = labels[b_ix, t_ix]
        seq_idx = b_ix.long()
        M = Q.size(0)
        V = self.item_vocab_size
        T = self.temperature
        mask_v = self.infonce_mask_value
        device = Q.device
        dtype = Q.dtype

        max_m = self.infonce_max_supervised
        if max_m > 0 and M > max_m:
            pick = torch.randperm(M, device=device)[:max_m]
            Q = Q[pick]
            pos_ids = pos_ids[pick]
            seq_idx = seq_idx[pick]
            M = Q.size(0)

        pos_emb = self.item_embedding(pos_ids)
        pos_logits = (Q * pos_emb).sum(dim=-1, keepdim=True) / T

        neg_ids = torch.randint(1, V, (M,), device=device, dtype=torch.long)
        eq = neg_ids == pos_ids
        for _ in range(8):
            if not eq.any():
                break
            r = torch.randint(1, V, (int(eq.sum().item()),), device=device, dtype=torch.long)
            neg_ids = neg_ids.clone()
            neg_ids[eq] = r
            eq = neg_ids == pos_ids
        Knegn = self.item_embedding(neg_ids)
        neg_logits = torch.mm(Q, Knegn.t()) / T

        invalid_easy = pos_ids.view(M, 1) == neg_ids.view(1, M)
        if invalid_easy.any():
            neg_logits = neg_logits.masked_fill(invalid_easy, mask_v)

        hard_logits = torch.mm(Q, pos_emb.t()) / T
        invalid_hard = seq_idx.view(M, 1) == seq_idx.view(1, M)
        if invalid_hard.any():
            hard_logits = hard_logits.masked_fill(invalid_hard, mask_v)

        logits = torch.cat([pos_logits, neg_logits, hard_logits], dim=1)
        target = torch.zeros(M, dtype=torch.long, device=device)
        return F.cross_entropy(logits, target)

    def _sampled_softmax_loss_flat(self, seq_flat, labels_flat):
        emb_w = self.item_embedding.weight
        mask = labels_flat > 0
        if not mask.any():
            return seq_flat.sum() * 0.0

        s = seq_flat[mask]
        y = labels_flat[mask]
        N = s.size(0)
        K = self.num_sampled_negs
        V = self.item_vocab_size

        if K < 1 or V <= 256:
            logits_full = torch.mm(s, emb_w.T)
            return F.cross_entropy(logits_full, y)

        pos_logits = (s * emb_w[y]).sum(dim=-1, keepdim=True)
        neg = torch.randint(1, V, (N, K), device=s.device, dtype=torch.long)
        eq = neg == y.unsqueeze(1)
        for _ in range(8):
            if not eq.any():
                break
            r = torch.randint(1, V, (int(eq.sum().item()),), device=s.device, dtype=torch.long)
            neg = neg.clone()
            neg[eq] = r
            eq = neg == y.unsqueeze(1)

        neg_emb = emb_w[neg]
        neg_logits = (s.unsqueeze(1) * neg_emb).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        target = torch.zeros(N, dtype=torch.long, device=s.device)
        return F.cross_entropy(logits, target)

    def forward(self, inputs):
        _, item_dict, mask = self.get_inputs(inputs)
        item_ids = item_dict[self.item_id_key].long()
        item_seq = item_ids[:, :-1]
        seq_emb = self._encode_sequence(item_seq)
        logits_last = self._scores_vs_item_weights(seq_emb[:, -1, :])
        return {"logits": logits_last, "y_pred": logits_last}

    def train_step(self, batch_data):
        _, item_dict, mask = self.get_inputs(batch_data)
        item_ids = item_dict[self.item_id_key].long()
        item_seq = item_ids[:, :-1]
        labels = item_ids[:, 1:].clone()
        labels[item_seq == 0] = 0

        seq_emb = self._encode_sequence(item_seq)

        if self.loss_type == "infonce_batch":
            loss = self._infonce_batch_loss(seq_emb, item_seq, labels)
        elif self.loss_type == "sampled_ce":
            seq_flat = seq_emb.reshape(-1, self.hidden_size)
            labels_flat = labels.reshape(-1)
            if (item_seq != 0).any():
                loss = self._sampled_softmax_loss_flat(seq_flat, labels_flat)
            else:
                loss = seq_flat.sum() * 0.0
        else:
            raise ValueError("loss_type must be 'infonce_batch' or 'sampled_ce', got {}".format(self.loss_type))

        loss = loss + self.regularization_loss()
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, return_dict, y_true):
        return F.cross_entropy(return_dict["logits"], y_true.long().view(-1)) + self.regularization_loss()

    def evaluate(self, data_generator, metrics=None):
        self.eval()
        total_loss = 0.0
        n_batches = 0
        total_hit = 0
        total_acc_cnt = 0
        with torch.no_grad():
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                _, item_dict, mask = self.get_inputs(batch_data)
                item_ids = item_dict[self.item_id_key].long()
                item_seq = item_ids[:, :-1]
                target = item_ids[:, -1]
                seq_lengths = mask.sum(dim=1).long()

                labels = item_ids[:, 1:].clone()
                labels[item_seq == 0] = 0
                seq_emb = self._encode_sequence(item_seq)

                if (item_seq != 0).any():
                    if self.loss_type == "infonce_batch":
                        batch_loss = self._infonce_batch_loss(seq_emb, item_seq, labels)
                    else:
                        seq_flat = seq_emb.reshape(-1, self.hidden_size)
                        labels_flat = labels.reshape(-1)
                        batch_loss = self._sampled_softmax_loss_flat(seq_flat, labels_flat)
                    total_loss += batch_loss.item()
                    n_batches += 1

                valid = (seq_lengths >= 1) & (target > 0)
                if valid.any():
                    last_h = seq_emb[valid, -1, :]
                    pred = self._argmax_predict_from_last_hidden(last_h)
                    total_hit += (pred == target[valid]).sum().item()
                    total_acc_cnt += int(valid.sum().item())

        mean_loss = total_loss / max(n_batches, 1)
        acc = total_hit / max(total_acc_cnt, 1)
        val_logs = {"logloss": mean_loss, "accuracy": acc}
        logging.info("[SASRecPretrain] " + " - ".join("{}: {:.6f}".format(k, v) for k, v in val_logs.items()))
        return val_logs

    def regularization_loss(self):
        reg = 0.0
        if self._net_regularizer:
            from fuxictr.pytorch.torch_utils import get_regularizer

            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "item_embedding" in name:
                    continue
                for net_p, net_lambda in net_reg:
                    reg += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        if self._embedding_regularizer:
            from fuxictr.pytorch.torch_utils import get_regularizer

            emb_reg = get_regularizer(self._embedding_regularizer)
            emb = self.item_embedding.weight
            if emb.requires_grad:
                for emb_p, emb_lambda in emb_reg:
                    reg += (emb_lambda / emb_p) * torch.norm(emb, emb_p) ** emb_p
        return reg


SASRec = SASRecPretrain
