import math
from typing import Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# huggingface配置类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = "silu",
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000,
            inference_rope_scaling: bool = False,
            flash_attention: bool = True,

            ############ MoE ############
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) * x

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

# 写出最初的RoPE公式
def precomput_freqs_cis(dim:int, end:int=int(32*1024), rope_base:float = 1e-6,
                        rope_scaling:Optional[dict] = None):
    # [:dim//2] 切片保证序列长度为dim//2，避免dim为奇数序列过长
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float()/dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        if end / orig_max > 1.0:
        # 计算corr_dim 这里所有的i是对二维组件的索引，范围是从0到dim//2-1。freqs[i]表示第i个二维组件的频率。
            corr_dim = next((i for i in range(dim//2) if 2*math.pi/ freqs[i] > orig_max), dim//2)

        # 计算power device=freqs.device是为了确保计算在相同设备上进行
            power = torch.arange(0, dim//2, device=freqs.device).float()/(max(dim//2-1,1))
        # 计算beta
            beta = beta_slow + (beta_fast - beta_slow) * power
        # 计算scale
            scale = torch.where(
                torch.arange(dim//2, device=freqs.device) < corr_dim,
                (beta*factor-beta+1)/(beta*factor),
                1.0/factor
            )
        # 应用scale
            freqs = freqs * scale

    # 生成位置索引，与频率相乘
    # 旋转角度θ = pos × freq × 2π 这里作用是批量计算θ的基数部分 也就是pos * freq
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # [end, dim//2]

# 返回一个cos和sin 用repeat_interleave方法交替重复 补充数据
#     freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [end, dim]
#     freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [end, dim]
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # 【seq_len,dim】
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # 【seq_len,dim】
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim =1):
    def rotate_half(x):
        # 正确逻辑：相邻维度拆分（偶数位、奇数位）
        # x = [q0,q1,q2,q3,q4,q5] → 偶数位：q0,q2,q4；奇数位：q1,q3,q5
        # 旋转后：[-q1,q0,-q3,q2,-q5,q4]
        return torch.cat([
            -x[..., 1::2],  # 取奇数位（1,3,5）并取负 → [-q1,-q3,-q5]
            x[..., ::2]  # 取偶数位（0,2,4）→ [q0,q2,q4]
        ], dim=-1)
    # 应用旋转位置编码
    # x_rotated = x * cos + rotate_half(x) * sin
    # unsqueeze用于后续的维度扩展
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        # 可能存在问题
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads # 查询头数量，本地注意力头数
        self.n_rep = self.n_local_heads // self.num_key_value_heads # 每个k/v头对应的查询头数量（也是复用次数）
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个注意力头的维度

        self.q_proj = nn.Linear(args.hidden_size,  args.num_attention_heads* self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size,  self.num_key_value_heads* self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size,  self.num_key_value_heads* self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads* self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self, x:torch.Tensor, position_embedding:Tuple[torch.Tensor, torch.Tensor],past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]],
                use_cache = False, attention_mask:Optional[torch.Tensor] = None) -> torch.Tensor:
    # 投影，计算qkv
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # [bsz, seq_len, n_heads*head_dim]
    # 把输入拆分成多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # [bsz, seq_len, n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)  # [bsz, seq_len, n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)  # [bsz, seq_len, n_kv_heads, head_dim]
    # q和k，使用RoPE
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len,:], sin[:seq_len,:])
    # 对于k和v，使用repeat，注意kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1,2),
            # [bsz, n_local_heads, seq_len, head_dim[
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2),
        )
    # 进行attention计算 若flash启动，则使用flash attention，为高性能版
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self, self.n_local_heads, seq_len, -1).bool()
            )
            # self.training 判断模型是否在训练模式下
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # 自回归（因果）注意力
            )
        else:
            # 计算注意力分数 q·k^T / sqrt(d)
            scores = (xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            socres = scores + torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            # 最后拼接头，输出投影，返回
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(scores)
            scores = self.attn_dropout(scores)
            output = scores@xv # @代表点乘
        # [bsz, n_local_heads, seq_len, head_dim]
        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            args.intermediate_size = int(args.hidden_size*8/3) # 论文中建议的升维比例
        # 保证intermediate_size是64的倍数 GPU的计算效率更高，非64倍数的维度会导致“内存碎片”
            args.intermediate_size = 64*((args.intermediate_size + 63)//64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act] # 获取激活函数

    def forward(self, x):
        # 门控+激活函数 使得梯度传递更平滑
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size # 总的隐藏层维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        # 层编号 用于区分不同block 推理时管理各层的KV Cache
        self.layer_id = layer_id
        # 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 实例化门控前馈层 FFN
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embedding, past_key_value=None,
                use_cache=False, attention_mask=None):
        residual = hidden_states # 残差连接 residual: [bsz, seq_len, hidden_size] batch size, 序列长度, 隐藏层维度
        # # 张量结构：2个样本 × 3个token × 4维特征
        # residual = [
        #     # 样本1（bsz=0）：3个token的特征向量
        #     [
        #         [0.1, 0.2, 0.3, 0.4],  # token0的4维特征
        #         [0.5, 0.6, 0.7, 0.8],  # token1的4维特征
        #         [0.9, 1.0, 1.1, 1.2]  # token2的4维特征
        #     ],
        #     # 样本2（bsz=1）：3个token的特征向量
        #     [
        #         [1.3, 1.4, 1.5, 1.6],  # token0的4维特征
        #         [1.7, 1.8, 1.9, 2.0],  # token1的4维特征
        #         [2.1, 2.2, 2.3, 2.4]  # token2的4维特征
        #     ]
        # ]
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # 注意力层输入先做RMSNorm（Pre-LN核心）
            position_embedding,  # RoPE位置编码（cos/sin）
            past_key_value,  # 历史KV Cache（推理时传入）
            use_cache=use_cache,  # 是否启用KV Cache（推理True/训练False）
            attention_mask=attention_mask,  # padding/因果掩码
        )
        hidden_states = residual + hidden_states # 残差连接
        # 先对注意力输出做RMSNorm，再传入MLP，最后残差相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # 返回当前Block的输出 + 更新后的KV Cache
        return hidden_states, present_key_value

class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层 将id变为向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE预计算
        freqs_cos, freqs_sin = precomput_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )
        # 注册缓冲区 跟着模型保存和加载
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, 'layers'):
            past_key_values = past_key_values.layers

        past_key_values = past_key_values or [None]*len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )
        presents = []

        for layer_idx, (layer, past_key_value) in enumerate (
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        self.config = config
        super().__init__(config)
        self.model = MokioMindModel(config)
        # 语言模型头 用于生成词汇分布 512层的隐藏层映射到6400词汇表
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 将词嵌入层和语言模型头的权重共享 让输出层的权重和嵌入层的权重共享 避免多计算一次weight
        self.model.embed_tokens.weight = self.lm_head.weight


    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                use_cache: bool = False,
                logits_to_keep:Union[int, torch.Tensor] = 0,
                **args
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        # logits to keep是整数，那就保留最后n个位置
        # 生成的时候只需要最后的logits来预测下一个token
        slice_indices = (slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep)
        logits = self.lm_head(hidden_states)[:, slice_indices, :]

        return CausalLMOutputWithPast(
            logits = logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states,
        )
