# =========================================================================
# SORT: ranking with optional frozen Phase-1 item ID embeddings (e.g. SASRec).
# Integrates with UniRank batch layout: (batch_dict, item_dict, mask[, multi_masks]).
# =========================================================================

import logging
import os

import torch
import torch.nn as nn
from typing import Union

from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.models import MultiTaskModel


def item_embedding_weight_from_checkpoint(
    checkpoint: Union[dict, torch.Tensor],
    weight_key: str = "item_embedding.weight",
) -> torch.Tensor:
    """
    Extract the item embedding matrix from torch.save output or a plain state_dict.
    Supports flat dicts and checkpoints that nest weights under model_state_dict / state_dict.
    """
    if torch.is_tensor(checkpoint):
        return checkpoint
    if weight_key in checkpoint:
        return checkpoint[weight_key]
    for inner_key in ("model_state_dict", "state_dict"):
        if inner_key in checkpoint and weight_key in checkpoint[inner_key]:
            return checkpoint[inner_key][weight_key]
    raise KeyError(
        "Cannot find {!r} in checkpoint (tried top-level and model_state_dict/state_dict).".format(
            weight_key
        )
    )


class SORT(MultiTaskModel):
    """
    Sequence: [BOS; History; SEP; User; SEP; Candidates] -> multi-task scores on candidates.
    Expects len(feature_map.labels) tasks; head outputs one logit per task (Sigmoid applied).
    """

    def __init__(
        self,
        feature_map,
        model_id="SORT",
        gpu=-1,
        item_id_key="item_index",
        pretrained_item_checkpoint=None,
        freeze_item_id_emb=True,
        item_id_emb_dim=64,
        model_dim=1280,
        embedding_dim=10,
        nhead=8,
        num_layers=6,
        transformer_dropout=0.1,
        padding_idx=0,
        learning_rate=1e-3,
        accumulation_steps=1,
        embedding_regularizer=None,
        net_regularizer=None,
        num_tasks=None,
        task=None,
        **kwargs,
    ):
        labels = feature_map.labels
        if num_tasks is None:
            num_tasks = len(labels)
        if num_tasks != len(labels):
            raise ValueError("num_tasks must match len(feature_map.labels)")

        if task is None:
            task = ["binary_classification"] * num_tasks
        if isinstance(task, str):
            task = [task] * num_tasks
        if len(task) != num_tasks:
            raise ValueError("task list length must match num_tasks")

        pretrained_item_checkpoint = kwargs.pop("pretrained_item_checkpoint", pretrained_item_checkpoint)
        weight_decay = float(kwargs.pop("weight_decay", 0) or 0)

        if "hidden_units" in kwargs:
            model_dim = int(kwargs.pop("hidden_units"))
        if "num_blocks" in kwargs:
            num_layers = int(kwargs.pop("num_blocks"))
        if "net_dropout" in kwargs:
            transformer_dropout = float(kwargs.pop("net_dropout"))
        if "num_heads" in kwargs:
            nhead = int(kwargs.pop("num_heads"))

        item_id_key = kwargs.pop("item_id_key", item_id_key)

        super(SORT, self).__init__(
            feature_map=feature_map,
            model_id=model_id,
            task=task,
            num_tasks=num_tasks,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )

        if item_id_key not in feature_map.features:
            raise KeyError(
                "item_id_key={!r} not in feature_map; set item_id_key to your item id column.".format(
                    item_id_key
                )
            )

        self.item_id_key = item_id_key
        self.padding_idx = int(padding_idx)
        self.accumulation_steps = accumulation_steps
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim

        if model_dim % nhead != 0:
            raise ValueError("model_dim must be divisible by nhead")

        vocab_size = int(feature_map.features[item_id_key]["vocab_size"])

        if pretrained_item_checkpoint:
            ckpt_path = os.path.abspath(os.path.expanduser(str(pretrained_item_checkpoint)))
            raw = torch.load(ckpt_path, map_location="cpu")
            w = item_embedding_weight_from_checkpoint(raw)
            if w.size(0) != vocab_size:
                logging.warning(
                    "Pretrained item embedding rows {} != feature_map vocab_size {}; check item_id_key / data.".format(
                        w.size(0), vocab_size
                    )
                )
            item_id_emb_dim = int(w.size(1))
            self.item_id_emb = nn.Embedding.from_pretrained(
                w,
                freeze=bool(freeze_item_id_emb),
                padding_idx=self.padding_idx,
            )
        else:
            if freeze_item_id_emb:
                logging.warning(
                    "pretrained_item_checkpoint is None; training item_id_emb from scratch (freeze_item_id_emb ignored)."
                )
            self.item_id_emb = nn.Embedding(vocab_size, int(item_id_emb_dim), padding_idx=self.padding_idx)

        self.user_emb_dim = 0
        self.item_aux_dim = 0
        for feat, spec in feature_map.features.items():
            if feat in feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            d = int(spec.get("embedding_dim", embedding_dim))
            src = spec.get("source")
            if src not in ("item", "action"):
                self.user_emb_dim += d
            elif feat != item_id_key:
                self.item_aux_dim += d

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        u_in = max(1, self.user_emb_dim)
        i_in = max(1, self.item_aux_dim)
        self.user_profile_proj = nn.Linear(u_in, model_dim)
        self.item_dense_proj = nn.Linear(i_in, model_dim)

        self.history_proj = nn.Linear(item_id_emb_dim + model_dim, model_dim)
        self.candidate_proj = nn.Linear(item_id_emb_dim + model_dim, model_dim)

        self.bos_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, num_tasks),
            nn.Sigmoid(),
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        if weight_decay > 0:
            opt = str(kwargs.get("optimizer", "adam")).lower()
            if opt == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif opt == "sgd":
                self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.reset_parameters()
        self.model_to_device()

    def _embed_user(self, batch_dict):
        feats = {}
        for k, v in batch_dict.items():
            if k in self.feature_map.labels:
                continue
            spec = self.feature_map.features[k]
            if spec.get("type") == "meta":
                continue
            if spec.get("source") in ("item", "action"):
                continue
            feats[k] = v
        bsz = next(iter(batch_dict.values())).size(0)
        dev = self.device
        if self.user_emb_dim == 0:
            return self.user_profile_proj(torch.ones(bsz, 1, device=dev, dtype=torch.float32))
        emb = self.embedding_layer(feats, flatten_emb=True)
        return self.user_profile_proj(emb)

    def _embed_item_aux(self, item_dict, seq_len):
        keys = [k for k in item_dict.keys() if k != self.item_id_key]
        aux = {k: item_dict[k] for k in keys}
        bsz = item_dict[self.item_id_key].size(0)
        dev = self.device
        if self.item_aux_dim == 0:
            flat = torch.ones(bsz * seq_len, 1, device=dev, dtype=torch.float32)
        else:
            flat = self.embedding_layer(aux, flatten_emb=True)
        return self.item_dense_proj(flat.view(bsz, seq_len, -1))

    def _encode_sequence(
        self,
        user_profile,
        hist_ids,
        hist_feats,
        cand_ids,
        cand_feats,
        hist_key_padding_mask,
        cand_key_padding_mask,
    ):
        batch_size = user_profile.size(0)
        device = user_profile.device

        hist_id_embs = self.item_id_emb(hist_ids)
        hist_tokens = self.history_proj(torch.cat([hist_id_embs, hist_feats], dim=-1))

        user_tokens = user_profile.unsqueeze(1)

        cand_id_embs = self.item_id_emb(cand_ids)
        cand_tokens = self.candidate_proj(torch.cat([cand_id_embs, cand_feats], dim=-1))

        bos = self.bos_token.expand(batch_size, -1, -1)
        sep = self.sep_token.expand(batch_size, -1, -1)

        sequence = torch.cat(
            [bos, hist_tokens, sep, user_tokens, sep, cand_tokens],
            dim=1,
        )

        false_b = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat(
            [
                false_b,
                hist_key_padding_mask,
                false_b,
                false_b,
                false_b,
                cand_key_padding_mask,
            ],
            dim=1,
        )

        out = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)
        num_candidates = cand_tokens.size(1)
        cand_out = out[:, -num_candidates:, :]
        return self.ranking_head(cand_out)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)

        if self.item_id_key not in item_dict:
            raise KeyError("item_dict missing item_id_key={!r}".format(self.item_id_key))

        item_ids = item_dict[self.item_id_key]
        batch_size, seq_len = item_ids.size()
        if seq_len < 2:
            raise ValueError("expects item sequence length >= 2 (history + at least one candidatei).")

        hist_ids = item_ids[:, :-1].long()
        cand_ids = item_ids[:, -1:].long()

        hist_pad = hist_ids == self.padding_idx
        if mask is not None:
            hist_pad = hist_pad | (mask == 0).bool()
        cand_pad = cand_ids == self.padding_idx

        item_side = self._embed_item_aux(item_dict, seq_len)
        hist_feats = item_side[:, :-1, :]
        cand_feats = item_side[:, -1:, :]

        user_profile = self._embed_user(batch_dict)

        predictions = self._encode_sequence(
            user_profile,
            hist_ids,
            hist_feats,
            cand_ids,
            cand_feats,
            hist_pad,
            cand_pad,
        )

        labels = self.feature_map.labels
        return_dict = {}
        for i, label in enumerate(labels):
            return_dict["{}_pred".format(label)] = predictions[:, :, i].reshape(-1, 1)
        return return_dict
