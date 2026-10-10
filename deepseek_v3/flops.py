TP_SIZE = 8
GB = 2**30
TB = 2**40
FLOAT16 = 2
FLOAT8 = 1
TFLOPS = 2**40

def RMSNorm(input_dim, input_shape):
    batch_size, seq_len, hidden_size = input_shape
    assert input_dim == hidden_size
    output_shape = input_shape
    flops = batch_size * seq_len * (3 * hidden_size + 2)
    return flops , output_shape

def ReplicatedLinear(input_dim, output_dim, input_shape):
    batch_size, seq_len, hidden_size = input_shape
    assert input_dim == hidden_size
    output_shape = (batch_size, seq_len, output_dim)
    flops = 2 * batch_size * seq_len * output_dim * input_dim
    return flops, output_shape

def ParallelLinear(input_dim, output_dim, input_shape, mode = 0):
    global TP_SIZE
    batch_size, seq_len, hidden_size = input_shape
    flops = 2 * batch_size * seq_len * input_dim * output_dim / TP_SIZE
    if mode == 0:
        assert input_dim == hidden_size
        output_shape = (batch_size, seq_len, output_dim/TP_SIZE)
    if mode == 1:
        assert input_dim/TP_SIZE  == hidden_size
        if TP_SIZE > 1:
            flops += 2 * seq_len * output_dim * batch_size
        output_shape = (batch_size, seq_len, output_dim)
    return flops, output_shape

def VocabParallelEmbedding(vocab_size, embedding_dim, input_shape):
    batch_size, seq_len = input_shape
    output_shape = (batch_size, seq_len, embedding_dim)
    return 0, output_shape

def DeepSeekV2MLP(hidden_size, intermiediate_size, input_shape):
    global TP_SIZE
    batch_size , seq_len, hidden_dim = input_shape
    assert hidden_dim == hidden_size
    flops = 0
    #up_proj
    flops0, out0_shape = ParallelLinear(hidden_size, intermediate_size * 2, input_shape)
    # silu and mul
    _, _, out0_dim = out0_shape
    assert out0_dim == intermediate_size/TP_SIZE*2
    flops1 = batch_size * 9 * seq_len * intermediate_size /TP_SIZE

    # down proj
    flops2, out1_shape = ParallelLinear(intermediate_size, hidden_size, (batch_size, seq_len, intermediate_size/TP_SIZE), mode=1)
    flops = flops0 + flops1 + flops2

    _, _, out1_dim = out1_shape
    assert out1_dim == hidden_size

    return flops, input_shape


def DeepSeekV2Moe(hidden_size,intermediate_size, n_routed_experts, num_experts, input_shape):
    global TP_SIZE
    num_experts = 8
    batch_size , seq_len, hidden_size = input_shape
    flops = 0
    #gate
    flops0 = 2 * batch_size * seq_len * hidden_size * n_routed_experts

    #shared
    flops1, shared_output_shape  = DeepSeekV2MLP(hidden_size, intermediate_size, input_shape)

    # experts
    flops2, experts_output_shape = DeepSeekV2MLP(hidden_size, moe_intermediate_size * num_experts, input_shape)

    #sum
    flops3 = batch_size * seq_len * hidden_size

    assert experts_output_shape == input_shape
    flops = flops0 + flops1 + flops2 + flops3
    return flops, input_shape

#https://github.com/vllm-project/vllm/blob/cd711c48b29a37f2bc4929bfe8291ab3107af505/vllm/model_executor/models/deepseek_v2.py#L323
def DeepseekV2MLAAttention(hidden_size,
                           num_attention_heads,
                           qk_nope_head_dim,
                           qk_rope_head_dim,
                           v_head_dim,
                           q_lora_rank,
                           kv_lora_rank,
                           max_position_embedding,
                           input_shape,
                           decode = False,
                           prefill = 0):
    global TP_SIZE
    global GB
    global FLOAT16
    global FLOAT8
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    num_heads = num_attention_heads
    num_local_heads = num_heads//TP_SIZE
    scaling = qk_head_dim**-0.5

    batch_size, seq_len, _hidden_size = input_shape
    assert _hidden_size == hidden_size

    flops = 0

    #q_a_proj
    flop0, ckq_shape = ReplicatedLinear(hidden_size, q_lora_rank , input_shape)

    #q_a_layernorm
    flop1, hidden_states_or_q_c_shape = RMSNorm(q_lora_rank, ckq_shape)

    #kv_a_proj_with_mqa
    flop2, k_cpe_shape = ReplicatedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim, input_shape)
    B, S, k_cpe_dim = k_cpe_shape
    assert k_cpe_dim == kv_lora_rank + qk_rope_head_dim
    kv_c_shape = (B, S, kv_lora_rank)
    k_pe_shape = (B, S, qk_rope_head_dim)

    #kv_a_layernorm
    flop3, kv_c_normed_shape = RMSNorm(kv_lora_rank, kv_c_shape)

    #self.mla_attn(hidden_states_or_q_c, kv_c_normed, k_pe)
    # hidden_states_or_q_c = b, s, q_lora_rank
    # kv_c_normed_shape = kv_c_shape = (B, S, kv_lora_rank)
    #k_pe_shape = (B, S, qk_rope_head_dim)
    flop4, q_b_proj_output_shape = ParallelLinear(q_lora_rank, num_heads * qk_head_dim, hidden_states_or_q_c_shape)

    flop5, kb_b_proj_output_shape = ParallelLinear(kv_lora_rank,
                                                        num_heads * (qk_nope_head_dim + v_head_dim),
                                                        kv_c_normed_shape)

    assert num_heads * qk_head_dim / TP_SIZE == q_b_proj_output_shape[2]
    assert num_heads * (qk_nope_head_dim + v_head_dim) / TP_SIZE == kb_b_proj_output_shape[2]

    if not decode:
        flop6 = batch_size * seq_len * seq_len * num_heads * (qk_nope_head_dim + qk_head_dim)/TP_SIZE * 2 \
                                        + batch_size * seq_len * num_heads * v_head_dim/TP_SIZE * seq_len * 2 \
                                        + batch_size * seq_len * seq_len * 7
    else:
        flop6 = batch_size * seq_len * (2 * prefill + seq_len + 1 ) * num_heads * (qk_nope_head_dim + qk_head_dim) /TP_SIZE \
                + batch_size * seq_len * (2 * prefill + seq_len + 1 ) * num_heads * v_head_dim/TP_SIZE \
                + batch_size * seq_len * seq_len/2 * (2 * prefill + seq_len + 1) * 7

    attention_output_shape  = (batch_size, seq_len, v_head_dim * num_heads/TP_SIZE)
    flop7, o_proj_output_shape = ParallelLinear(v_head_dim * num_heads, hidden_size,  attention_output_shape, mode=1)

    flops = flop0 + flop1 + flop2 + flop3 + flop4 + flop5 + flop6 + flop7

    assert o_proj_output_shape == (batch_size, seq_len, hidden_size)
    return flops, o_proj_output_shape

def LMHead(hidden_size, vocab_size, input_shape):
    global TP_SIZE
    batch_size, seq_len, hidden_dim = input_shape
    assert hidden_size == hidden_dim

    flops = 2 * seq_len * hidden_size * vocab_size/TP_SIZE
    out_shape = (batch_size, seq_len, vocab_size/TP_SIZE)
    return flops, out_shape

if __name__ == "__main__":
    # config
    import argparse
    parser = argparse.ArgumentParser(description="Process batch, prefill length, and decode length.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--prefill-len", type=int, default=1, help="Prefill length")
    parser.add_argument("--decode-len", type=int, default=1, help="Decode length")

    args = parser.parse_args()

    print(f"BATCH = {args.batch}")
    print(f"PREFILL_LEN = {args.prefill_len}")
    print(f"DECODE_LEN = {args.decode_len}")

    args = parser.parse_args()
    hidden_size = 7168
    intermediate_size = 18432
    kv_lora_rank = 512
    max_position_embedding = 163840
    moe_intermediate_size = 2048
    n_group = 8
    n_routed_experts = 256
    n_shared_experts = 1
    num_attention_heads =  128
    num_experts_per_token = 8
    num_hidden_layers = 61
    num_key_value_head = 128
    q_lora_rank = 1536
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    vocab_size = 129280
    topk_group = 4



    BATCH = args.batch
    PREFILL_LEN = args.prefill_len
    DECODE_LEN = args.decode_len

    prefill_flops = 0
    decode_flops = 0

    #Prefill
    batch_size = BATCH
    seq_len = PREFILL_LEN
    input = (batch_size, seq_len)
    prefill_flops = 0
    flop, embeded_tokens = VocabParallelEmbedding(vocab_size, hidden_size, input)
    prefill_flops += flop


    for i in range(3):
        flop0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
        prefill_flops += flop0
        flop1, attn_out = DeepseekV2MLAAttention(hidden_size,
                            num_attention_heads,
                            qk_nope_head_dim,
                            qk_rope_head_dim,
                            v_head_dim,
                            q_lora_rank,
                            kv_lora_rank,
                            max_position_embedding,
                            input_layernorm)
        prefill_flops += flop1
        flop2, output_layernorm = RMSNorm(hidden_size, attn_out)
        prefill_flops += flop2
        flop3, mlp_out = DeepSeekV2MLP(hidden_size, intermediate_size, output_layernorm)
        prefill_flops += flop3

    for i in range(num_hidden_layers - 3):
        flop0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
        prefill_flops += flop0
        flop1, attn_out = DeepseekV2MLAAttention(hidden_size,
                            num_attention_heads,
                            qk_nope_head_dim,
                            qk_rope_head_dim,
                            v_head_dim,
                            q_lora_rank,
                            kv_lora_rank,
                            max_position_embedding,
                            input_layernorm)
        prefill_flops += flop1
        flop2, output_layernorm = RMSNorm(hidden_size, attn_out)
        prefill_flops += flop2
        flop3, mlp_out = DeepSeekV2Moe(hidden_size, intermediate_size,  n_routed_experts, num_experts_per_token, output_layernorm)
        prefill_flops += flop3

    flop_layernorm, output_layernorm = RMSNorm(hidden_size, mlp_out)
    prefill_flops += flop_layernorm

    #lm head
    prefill_flops_head, output_head = LMHead(hidden_size, vocab_size, mlp_out)
    prefill_flops += prefill_flops_head

    if DECODE_LEN > 1:
        batch_size = BATCH
        seq_len = DECODE_LEN - 1
        input = (batch_size, seq_len)


        decode_flops = 0
        flop, embeded_tokens = VocabParallelEmbedding(vocab_size, hidden_size, input)
        decode_flops += flop


        for i in range(3):
            flop0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
            decode_flops += flop0
            flop1, attn_out = DeepseekV2MLAAttention(hidden_size,
                                num_attention_heads,
                                qk_nope_head_dim,
                                qk_rope_head_dim,
                                v_head_dim,
                                q_lora_rank,
                                kv_lora_rank,
                                max_position_embedding,
                                input_layernorm,
                                decode=True,
                                prefill=PREFILL_LEN)
            decode_flops += flop1
            flop2, output_layernorm = RMSNorm(hidden_size, attn_out)
            decode_flops += flop2
            flop3, mlp_out = DeepSeekV2MLP(hidden_size, intermediate_size, output_layernorm)
            decode_flops += flop3

        for i in range(num_hidden_layers - 3):
            flop0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
            decode_flops += flop0
            flop1, attn_out = DeepseekV2MLAAttention(hidden_size,
                                num_attention_heads,
                                qk_nope_head_dim,
                                qk_rope_head_dim,
                                v_head_dim,
                                q_lora_rank,
                                kv_lora_rank,
                                max_position_embedding,
                                input_layernorm,
                                decode=True,
                                prefill=PREFILL_LEN)
            decode_flops += flop1
            flop2, output_layernorm = RMSNorm(hidden_size, attn_out)
            decode_flops += flop2
            flop3, mlp_out = DeepSeekV2Moe(hidden_size, intermediate_size,  n_routed_experts, num_experts_per_token, output_layernorm)
            decode_flops += flop3

        flop_layernorm, output_layernorm = RMSNorm(hidden_size, mlp_out)
        decode_flops += flop_layernorm

        #lm head
        decode_flops_head, output_head = LMHead(hidden_size, vocab_size, mlp_out)
        decode_flops += decode_flops_head



    print("Prefill FLOPs: ", prefill_flops/TFLOPS)
    print("Deocde FLOPs: ", decode_flops/TFLOPS)
    print("TOTAL FLOPs: ", (prefill_flops + decode_flops)/TFLOPS)
