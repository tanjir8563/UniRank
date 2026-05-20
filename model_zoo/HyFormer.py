import math
import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, Dice, MaskedAveragePooling, MultiHeadTokenMixing, PerTokenFeedForward, ScaledDotProductAttention
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist

class HyFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="HyFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=3,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 num_global_token=1,
                 sequence_encoder_type="transformer",
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(HyFormer, self).__init__(feature_map,
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
        self.num_global_token = num_global_token
        self.accumulation_steps = accumulation_steps
        self.masked_avg_pooling = MaskedAveragePooling()

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

        # 非序列特征 + target item -> NS tokens
        self.ns_tokenizer = Embedding2Tokenization(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=token_dim,
            num_ns_token=num_ns_token,
            dnn_activations=dnn_activations
        )

        # Query Generation:
        # Global Info = concat(non-seq features, target item, meanpool(sequence))
        self.query_generator = QueryGeneration(
            input_dim=self.non_item_dim + self.item_info_dim + token_dim,
            token_dim=token_dim,
            num_global_token=num_global_token,
            dnn_activations=dnn_activations
        )

        # item sequence / target item 投影到统一 token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # HyFormer 主体
        self.unified_layers = HyFormerBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=num_ns_token,
            num_global_token=num_global_token,
            dnn_activations=dnn_activations,
            sequence_encoder_type=sequence_encoder_type
        )

        # 最终使用 Global Tokens 做预测
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
        batch_size = mask.shape[0]

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_feat_emb[:, 0:-1, :]  # B x T x item_info_dim

        # sequence tokens Input Tokenization
        s_tokens = self.item_token_proj(sequence_emb)  # B x T x token_dim
        s_tokens = s_tokens * mask.unsqueeze(-1).float()

        # non-seq embedding
        batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # B x non_item_dim

        # NS tokens Input Tokenization
        ns_input = torch.cat([batch_emb, target_emb], dim=-1)
        ns_tokens = self.ns_tokenizer(ns_input)  # B x num_ns_token x token_dim

        # Global Tokens from Query Generation
        seq_pool = self.masked_avg_pooling(s_tokens, mask)  # B x token_dim
        global_info = torch.cat([batch_emb, target_emb, seq_pool], dim=-1)
        global_tokens = self.query_generator(global_info)  # B x num_global_token x token_dim

        # HyFormer block
        _, global_tokens, _ = self.unified_layers(
            s_tokens, global_tokens, ns_tokens, mask
        )

        # final prediction
        bottom_output = global_tokens.mean(dim=1)

        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class Embedding2Tokenization(nn.Module):
    def __init__(self, input_dim, token_dim, num_ns_token, dnn_activations="ReLU"):
        super(Embedding2Tokenization, self).__init__()
        self.num_ns_token = num_ns_token
        self.token_dim = token_dim
        output_dim = token_dim * num_ns_token
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            get_activation(dnn_activations),
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


class QueryGeneration(nn.Module):
    """
    Global Info = Concat(non-seq features, target item, pooled sequence summary)
    Q0 = MLP(Global Info)
    """
    def __init__(self, input_dim, token_dim, num_global_token, dnn_activations="ReLU"):
        super(QueryGeneration, self).__init__()
        self.num_global_token = num_global_token
        self.token_dim = token_dim
        output_dim = token_dim * num_global_token
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            get_activation(dnn_activations),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, global_info):
        """
        global_info: B x input_dim
        return: B x num_global_token x token_dim
        """
        q = self.mlp(global_info)
        q = q.view(-1, self.num_global_token, self.token_dim)
        return q


class HyFormerBlock(nn.Module):
    """
    每层:
    1) Sequence Representation Encoding
    2) Query Decoding: global tokens -> cross attend over sequence tokens
    3) Query Boosting: concat(decoded global tokens, ns tokens) -> mixer -> residual
    """
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 num_global_token,
                 dnn_activations="ReLU",
                 sequence_encoder_type="transformer"):
        super(HyFormerBlock, self).__init__()
        self.num_layers = num_layers
        self.num_global_token = num_global_token

        self.seq_layers = nn.ModuleList([
            SequenceRepresentationLayer(
                input_dim=input_dim,
                num_heads=num_heads,
                dnn_activations=dnn_activations,
                sequence_encoder_type=sequence_encoder_type
            ) for _ in range(num_layers)
        ])

        self.decode_layers = nn.ModuleList([
            QueryDecodingLayer(
                input_dim=input_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)
        ])

        self.boost_layers = nn.ModuleList([
            QueryBoostingLayer(
                input_dim=input_dim,
                num_global_token=num_global_token,
                num_ns_token=num_ns_token,
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, global_tokens, ns_tokens, mask=None):
        """
        s_tokens:      B x Ls x D
        global_tokens: B x G  x D
        ns_tokens:     B x M  x D
        """
        G = global_tokens.size(1)

        for i in range(self.num_layers):
            # 1) Sequence Representation Encoding
            s_tokens = self.seq_layers[i](s_tokens, mask)

            # 2) Query Decoding
            decoded_global = self.decode_layers[i](global_tokens, s_tokens, mask)

            # 3) Query Boosting
            fusion_tokens = torch.cat([decoded_global, ns_tokens], dim=1)   # B x (G+M) x D
            boosted_tokens = self.boost_layers[i](fusion_tokens)

            global_tokens = boosted_tokens[:, :G, :]
            ns_tokens = boosted_tokens[:, G:, :]

        return s_tokens, global_tokens, ns_tokens


class SequenceRepresentationLayer(nn.Module):
    """
    对应论文中的 Sequence Representation Encoding。
    - 这里只复现fine-grained interactions的Full Transformer Encoding，即标准 self-attention + FFN
    - trade-off：没有直接在scaled_dot_product_attention引入自定义mask，用少量的注意力权重稀释换取FlashAttention的计算加速
    """
    def __init__(self, input_dim, num_heads, dnn_activations="ReLU", sequence_encoder_type="transformer"):
        super(SequenceRepresentationLayer, self).__init__()
        self.sequence_encoder_type = sequence_encoder_type
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(input_dim)
        else:
            self.norm = nn.LayerNorm(input_dim)

        if sequence_encoder_type == "transformer":
            self.q_proj = nn.Linear(input_dim, input_dim)
            self.k_proj = nn.Linear(input_dim, input_dim)
            self.v_proj = nn.Linear(input_dim, input_dim)
            self.out_proj = nn.Linear(input_dim, input_dim)
        else:
            raise ValueError("sequence_encoder_type not implemented")

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            get_activation(dnn_activations),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, s_tokens, mask=None):

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()
        x = s_tokens

        norm_x = self.norm(x)
        B, L, D = norm_x.shape

        # Q, K, V projections -> (B, num_heads, L, head_dim)
        q = self.q_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        x = attn_out + x
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        x = self.ffn(x) + x
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class QueryDecodingLayer(nn.Module):
    """
    Q^(l) = CrossAttn(Q^(l-1), K^(l), V^(l))
    工程上这里加了 pre-norm + residual，训练更稳。
    - trade-off：没有直接在scaled_dot_product_attention引入自定义mask，用少量的注意力权重稀释换取FlashAttention的计算加速，
      如果在scaled_dot_product_attention直接引入自定义mask，反而效果降低
    """
    def __init__(self, input_dim, num_heads):
        super(QueryDecodingLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(input_dim)
        else:
            self.norm = nn.LayerNorm(input_dim)

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, global_tokens, s_tokens, mask=None):
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        norm_q = self.norm(global_tokens)
        norm_kv = self.norm(s_tokens)

        B, Lq, D = norm_q.shape
        Lk = norm_kv.shape[1]

        # Q from global tokens, K/V from sequence tokens
        q = self.q_proj(norm_q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        attn_out = self.out_proj(attn_out)

        return attn_out + global_tokens


class QueryBoostingLayer(nn.Module):
    """
    论文里的 Query Boosting，实际上就是RankMixer，也可以尝试使用MLP-Mixer代替MultiHeadTokenMixing
    """
    def __init__(self, input_dim, num_global_token, num_ns_token):
        super(QueryBoostingLayer, self).__init__()
        self.num_tokens = num_global_token + num_ns_token
        self.input_dim = input_dim
        if hasattr(nn, "RMSNorm"):
            self.mixer_norm = nn.RMSNorm(input_dim)
            self.pffn_norm = nn.RMSNorm(input_dim)
        else:
            self.mixer_norm = nn.LayerNorm(input_dim)
            self.pffn_norm = nn.LayerNorm(input_dim)
        self.token_mixer = MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token + num_global_token)
        self.pffn = PerTokenFeedForward(input_dim=input_dim, num_token=num_ns_token + num_global_token)

    def forward(self, x: torch.Tensor):
        x = self.token_mixer(self.mixer_norm(x)) + x
        x = self.pffn(self.pffn_norm(x)) + x
        return x