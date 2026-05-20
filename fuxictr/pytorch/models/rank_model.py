# =========================================================================
# Copyright (C) 2026. The UniRank Library. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch.nn as nn
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import os, sys
import logging
from fuxictr.pytorch.layers import FeatureEmbeddingDict
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm
from contextlib import nullcontext

try:
    from torch.distributed.algorithms.join import Join
except Exception:
    Join = None

class BaseModel(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="BaseModel",
                 task="binary_classification",
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps  # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]

        # DDP related
        self.distributed = kwargs.get("distributed", False)
        self.rank = kwargs.get("rank", 0)
        self.local_rank = kwargs.get("local_rank", 0)
        self.world_size = kwargs.get("world_size", 1)
        self._ddp_model = None

        # bf16 混合精度开关
        self.enable_bf16 = kwargs.get("enable_bf16", True)
        self.enable_torch_compile = kwargs.get("enable_torch_compile", True)
        self._torch_compile_enabled = False

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)
        self._maybe_enable_torch_compile()

    def _maybe_enable_torch_compile(self):
        if self._torch_compile_enabled or (not self.enable_torch_compile):
            return
        if not hasattr(torch, "compile"):
            logging.warning("torch.compile is not available in this PyTorch build, skip compile().")
            return
        if not hasattr(self, "unified_layers"):
            logging.warning("No unified_layers found on model, skip compile().")
            return
        try:
            logging.info("************ compile start ************")
            self.unified_layers = torch.compile(
                self.unified_layers,
                backend="inductor",
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
                fullgraph=False,
            )
            self._torch_compile_enabled = True
        except Exception as ex:
            logging.warning("torch.compile failed and will be skipped: %s", ex)

    def set_ddp_model(self, ddp_model):
        """Attach DDP wrapper for forward/backward only, without module registration cycle."""
        if "_ddp_model" in self._modules:
            self._modules.pop("_ddp_model")
        object.__setattr__(self, "_ddp_model", ddp_model)

    def _is_distributed(self):
        return self.distributed and dist.is_available() and dist.is_initialized()

    def _is_main_process(self):
        return (not self._is_distributed()) or (self.rank == 0)

    def _train_forward_model(self):
        return self._ddp_model if self._ddp_model is not None else self

    def _get_amp_context(self):
        """
        返回 bf16 autocast 上下文管理器。
        - enable_bf16=True  且设备为 CUDA：使用 torch.autocast(cuda, bfloat16)
        - enable_bf16=True  且设备为 CPU ：使用 torch.autocast(cpu,  bfloat16)
        - enable_bf16=False：返回 nullcontext()，不做任何精度转换
        """
        if self.enable_bf16:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _sync_stop_flag_from_main(self):
        """Broadcast early-stop flag from rank0 to all ranks."""
        if not self._is_distributed():
            return
        if self.device.type == "cuda":
            flag = torch.tensor([1 if self._stop_training else 0], device=self.device, dtype=torch.int32)
        else:
            flag = torch.tensor([1 if self._stop_training else 0], dtype=torch.int32)
        dist.broadcast(flag, src=0)
        self._stop_training = bool(flag.item())

    def _sync_lr_from_main(self):
        """Broadcast learning rate from rank0 to all ranks."""
        if not self._is_distributed():
            return
        lr_tensor = torch.tensor(
            [self.optimizer.param_groups[0]["lr"]],
            device=self.device, dtype=torch.float64
        )
        dist.broadcast(lr_tensor, src=0)
        new_lr = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            emb_params = set()
            for m_name, module in self.named_modules():
                if type(module) == FeatureEmbeddingDict:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            emb_params.add(".".join([m_name, p_name]))
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in emb_params:
                        for net_p, net_lambda in net_reg:
                            reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()

        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None, return_multi_masks=False):
        """
        Args:
            inputs: (batch_dict, item_dict, mask) 或 (batch_dict, item_dict, mask, multi_masks)
            feature_source: 可选特征来源过滤
            return_multi_masks: bool, 默认 False
                - False: 返回 (X_dict, item_dict, mask)
                - True : 返回 (X_dict, item_dict, mask, multi_masks)
        """
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

        mask = mask.to(self.device)

        if not return_multi_masks:
            return X_dict, item_dict, mask

        if multi_masks is None:
            multi_masks = [mask for _ in range(self.num_tasks)]
        else:
            if isinstance(multi_masks, (list, tuple)):
                multi_masks = [m.to(self.device) for m in multi_masks]
            elif torch.is_tensor(multi_masks):
                multi_masks = multi_masks.to(self.device)
                if multi_masks.dim() == 3 and multi_masks.size(0) == self.num_tasks:
                    multi_masks = [multi_masks[i] for i in range(self.num_tasks)]
                else:
                    multi_masks = [multi_masks for _ in range(self.num_tasks)]
            else:
                multi_masks = [mask for _ in range(self.num_tasks)]
        return X_dict, item_dict, mask, multi_masks

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        """
        DDP + blocked dataloader 训练说明：
        - blocked=True 时，不同 rank 的本地 batch 数可能不完全一致；
        - DDP 训练阶段如果某些 rank 先进入 eval/barrier，而其他 rank 还在 backward allreduce，
          会导致 NCCL collective 顺序不一致并最终 timeout；
        - 因此在 DDP + blocked 模式下，强制禁止 step 内 eval，只在所有 rank 完成当前 epoch 后统一 eval；
        - 同时用 torch.distributed.algorithms.join.Join 处理 uneven inputs，让先结束的 rank
          shadow 后续 collective，避免最后几个训练 batch 卡死。
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0

        # 当前 dataloader 是否是 blocked 模式。
        # UniRankDataloader 在 __init__ 中会设置 self.blocked。
        self._blocked_training = bool(getattr(data_generator, "blocked", False))

        # DDP + blocked 下强制 epoch-end eval only，避免各 rank 因本地 len(data_generator)
        # 不同而在不同 step 进入 eval，导致 collective 顺序错乱。
        self._epoch_end_eval_only = bool(self._is_distributed() and self._blocked_training)

        if self._eval_steps is None:
            # 原始语义：None 表示每个 epoch 验证一次。
            # 在非 blocked 场景下仍然保留 train_epoch 内部按 _steps_per_epoch 触发的逻辑；
            # 在 DDP + blocked 场景下，train_epoch 内部会跳过 eval，fit() 在 epoch 末统一 eval。
            self._eval_steps = self._steps_per_epoch

        if self._is_main_process():
            logging.info("BF16 mixed precision: {}".format(self.enable_bf16))
            logging.info("Start training: {} local batches/epoch".format(self._steps_per_epoch))
            logging.info("DDP blocked training: {}".format(self._epoch_end_eval_only))
            if self._epoch_end_eval_only:
                logging.info("Disable step-wise eval and use epoch-end synchronized eval for DDP + blocked dataloader.")
            logging.info("************ Epoch=1 start ************")

        for epoch in range(epochs):
            self._epoch_index = epoch

            # DDP: make DistributedSampler / blocked iterable shuffle differently every epoch
            if hasattr(data_generator, "set_epoch"):
                data_generator.set_epoch(epoch)
            elif hasattr(data_generator, "sampler") and hasattr(data_generator.sampler, "set_epoch"):
                data_generator.sampler.set_epoch(epoch)

            use_ddp_join = bool(
                self._is_distributed()
                and self._ddp_model is not None
                and Join is not None
                and self._blocked_training
            )

            if use_ddp_join:
                # uneven inputs 场景：某些 rank 先耗尽数据时，Join 会让它们继续 shadow
                # 其他 rank 后续 backward 中的 collective，避免最后几个 allreduce 卡住。
                with Join([self._ddp_model], throw_on_early_termination=False):
                    epoch_loss, epoch_batches = self.train_epoch(data_generator)
            else:
                if self._is_distributed() and self._blocked_training and Join is None:
                    logging.warning(
                        "torch.distributed.algorithms.join.Join is unavailable. "
                        "DDP + blocked with uneven local batches may still hang."
                    )
                epoch_loss, epoch_batches = self.train_epoch(data_generator)

            # DDP + blocked: 只允许所有 rank 完成训练 epoch 后统一 eval。
            if self._epoch_end_eval_only and (not self._stop_training):
                if self._is_main_process():
                    denom = max(1, epoch_batches)
                    logging.info("Train loss: {:.6f}".format(epoch_loss / denom))

                if self._is_distributed():
                    dist.barrier()

                self.eval_step()

                if self._is_distributed():
                    dist.barrier()
                    self._sync_stop_flag_from_main()
                    self._sync_lr_from_main()

            if self._stop_training:
                break
            else:
                if self._is_main_process():
                    logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
                    if epoch + 1 < epochs:
                        logging.info("************ Epoch={} start ************".format(self._epoch_index + 2))

        if self._is_main_process():
            logging.info("Training finished.")
            logging.info("Load best model: {}".format(self.checkpoint))

        # Ensure checkpoint is fully written before all ranks load
        if self._is_distributed():
            dist.barrier()
        self.load_weights(self.checkpoint)
        if self._is_distributed():
            dist.barrier()

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        # Only rank0 decides early-stop and saves checkpoints
        if not self._is_main_process():
            return

        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
                (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}" \
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        # 所有 rank 都参与验证推理，通过 all_gather 汇聚后由 rank 0 计算指标
        if self._is_main_process():
            logging.info('Evaluation @epoch {} - batch {}: '.format(
                self._epoch_index + 1, self._batch_index + 1))

        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())

        if self._is_main_process():
            self.checkpoint_and_earlystop(val_logs)

        self.train()  # 所有 rank 都需要恢复 train 模式

    def train_step(self, batch_data):
        is_update_step = ((self._batch_index + 1) % self.accumulation_steps == 0)
        use_no_sync = (
                self._is_distributed()
                and (self._ddp_model is not None)
                and self.accumulation_steps > 1
                and (not is_update_step)
        )
        sync_ctx = self._ddp_model.no_sync() if use_no_sync else nullcontext()
        amp_ctx = self._get_amp_context()

        with sync_ctx:
            # 仅前向传播在 amp_ctx 内，享受 bf16 加速
            with amp_ctx:
                return_dict = self._train_forward_model()(batch_data)

            # 退出 autocast 后，将所有浮点预测值转回 float32，
            # 避免 BCELoss / binary_cross_entropy 在 bf16 下报错
            return_dict = {
                k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in return_dict.items()
            }

            y_true = self.get_labels(batch_data)
            loss = self.compute_loss(return_dict, y_true) / self.accumulation_steps
            loss.backward()

        if is_update_step:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0.0
        num_batches = 0
        self.train()
        self.optimizer.zero_grad()

        # 每个 rank 都可以显示 tqdm
        use_tqdm = (self._verbose > 0)
        if use_tqdm:
            batch_iterator = tqdm(
                data_generator,
                disable=False,
                file=sys.stdout,
                desc=f"Rank {self.rank} | Epoch {self._epoch_index + 1}",
                position=self.rank,
                leave=(self.rank == 0),
                dynamic_ncols=True
            )
        else:
            batch_iterator = data_generator

        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            num_batches += 1

            loss = self.train_step(batch_data)
            loss_value = loss.item() * self.accumulation_steps
            train_loss += loss_value

            if use_tqdm:
                avg_loss = train_loss / max(1, num_batches)
                batch_iterator.set_postfix(
                    loss=f"{loss_value:.6f}",
                    avg_loss=f"{avg_loss:.6f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # DDP + blocked 场景下，禁止训练阶段 step 内 eval。
            # 原因：blocked dataloader 各 rank 本地 batch 数可能不同，某个 rank 先进入 eval
            # 而另一个 rank 还在 backward allreduce，会导致 NCCL collective 顺序不一致。
            do_step_eval = (
                (not getattr(self, "_epoch_end_eval_only", False))
                and self._eval_steps is not None
                and self._eval_steps > 0
                and self._total_steps % self._eval_steps == 0
            )

            if do_step_eval:
                if self._is_main_process():
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0.0

                self.eval_step()

                if self._is_distributed():
                    dist.barrier()
                    self._sync_stop_flag_from_main()
                    self._sync_lr_from_main()

            if self._stop_training:
                break

        # ---- flush 残留梯度 ----
        # 注意：如果使用 no_sync 做梯度累积，最后不足 accumulation_steps 的残留梯度
        # 在原始实现中会直接 optimizer.step()，这些梯度可能没有经历 DDP allreduce。
        # 为了保持行为兼容，这里仍保留原逻辑；更严格的做法是让 dataloader/drop_last
        # 或训练步数保证每次 update 都完整对齐。
        if self.accumulation_steps > 1 and num_batches > 0 and ((self._batch_index + 1) % self.accumulation_steps != 0):
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return train_loss, num_batches

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            raw_generator = data_generator
            y_pred = []
            y_true = []
            group_id = []

            # 改动：每个 rank 都显示自己的验证进度
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator,
                    disable=False,
                    file=sys.stdout,
                    desc=f"Rank {self.rank} | Eval",
                    position=self.rank,
                    leave=(self.rank == 0),
                    dynamic_ncols=True
                )

            amp_ctx = self._get_amp_context()
            with amp_ctx:
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                    if self.feature_map.group_id is not None:
                        group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))

            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            # ---- 分布式一致性检查：所有 rank 必须对"是否做分布式评估聚合"判断一致 ----
            _distributed_eval = False
            _is_sampler_distributed_eval = False
            _is_blocked_distributed_eval = False

            if self._is_distributed():
                _is_sampler_distributed_eval = bool(
                    hasattr(raw_generator, "sampler")
                    and isinstance(raw_generator.sampler, DistributedSampler)
                )

                _is_blocked_distributed_eval = bool(
                    getattr(raw_generator, "blocked", False)
                    and hasattr(raw_generator, "dataset")
                    and getattr(raw_generator.dataset, "distributed", False)
                )

                local_flag = int(_is_sampler_distributed_eval or _is_blocked_distributed_eval)

                flag_t = torch.tensor([local_flag], dtype=torch.int32, device=self.device)
                gathered_flags = [torch.zeros_like(flag_t) for _ in range(self.world_size)]
                dist.all_gather(gathered_flags, flag_t)
                flags = [int(x.item()) for x in gathered_flags]

                if len(set(flags)) != 1:
                    raise RuntimeError(
                        f"[Rank {self.rank}] Inconsistent distributed-eval flags across ranks: {flags}. "
                        "这会导致 collective 死锁。请确保各 rank 的验证 DataLoader 配置一致。"
                    )

                _distributed_eval = bool(flags[0])

                if self.force_distributed_eval and not _distributed_eval:
                    raise RuntimeError(
                        "force_distributed_eval=True, 但当前评估 DataLoader 未启用分布式切分。"
                    )

            # ---- 分布式评估聚合 ----
            if _distributed_eval:
                if _is_blocked_distributed_eval:
                    local_samples_t = torch.tensor([len(y_true)], dtype=torch.int64, device=self.device)
                    dist.all_reduce(local_samples_t, op=dist.ReduceOp.SUM)
                    total_samples = int(local_samples_t.item())
                else:
                    total_samples = len(raw_generator.dataset)

                y_pred, y_true, group_id = self._gather_eval_results(
                    y_pred, y_true, group_id, total_samples
                )

                # 非主进程不算指标，直接返回空字典
                if not self._is_main_process():
                    return OrderedDict()

            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)

            if self._is_main_process():
                logging.info('[Metrics] ' + ' - '.join(
                    '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))

            return val_logs

    def _gather_eval_results(self, y_pred, y_true, group_id, total_samples):
        """
        更稳健的分布式评估聚合：
        - 优先 gather_object 仅汇总到 rank0（避免 GPU all_gather 大张量）
        - rank0 拼接后裁剪到 total_samples（去除 DistributedSampler padding）
        """
        payload = {
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "group_id": (None if group_id is None else np.asarray(group_id))
        }

        # 仅 rank0 收集，降低内存和通信压力
        if hasattr(dist, "gather_object"):
            if self._is_main_process():
                gathered = [None for _ in range(self.world_size)]
                dist.gather_object(payload, gathered, dst=0)
            else:
                dist.gather_object(payload, None, dst=0)
                return None, None, None
        else:
            # 兜底：所有 rank 都收（老版本 torch）
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, payload)
            if not self._is_main_process():
                return None, None, None

        y_pred = np.concatenate([g["y_pred"] for g in gathered], axis=0)[:total_samples]
        y_true = np.concatenate([g["y_true"] for g in gathered], axis=0)[:total_samples]

        has_gid = any(g["group_id"] is not None for g in gathered)
        if has_gid:
            if not all(g["group_id"] is not None for g in gathered):
                raise RuntimeError("Some ranks have group_id while others do not.")
            group_id = np.concatenate([g["group_id"] for g in gathered], axis=0)[:total_samples]
        else:
            group_id = None

        return y_pred, y_true, group_id

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            # 改动：每个 rank 都显示自己的预测进度
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator,
                    disable=False,
                    file=sys.stdout,
                    desc=f"Rank {self.rank} | Predict",
                    position=self.rank,
                    leave=(self.rank == 0),
                    dynamic_ncols=True
                )

            amp_ctx = self._get_amp_context()
            with amp_ctx:
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        embedding_params = 0
        dense_params = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            param_num = param.numel()

            if "embedding" in name:
                embedding_params += param_num
                if count_embedding:
                    total_params += param_num
            else:
                dense_params += param_num
                total_params += param_num

        if self._is_main_process():
            logging.info("Total number of parameters: {}.".format(total_params))
            logging.info("Number of embedding parameters: {}.".format(embedding_params))
            logging.info("Number of dense parameters: {}.".format(dense_params))