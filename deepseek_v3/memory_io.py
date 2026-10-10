TP_SIZE = 8
GB = 2**30
TB = 2**40
FLOAT16 = 2
FLOAT8 = 1

def RMSNorm(input_dim, input_shape, decode = False):
    batch_size, seq_len, hidden_size = input_shape
    assert input_dim == hidden_size
    output_shape = input_shape
    if decode:
        mem_io = batch_size * (seq_len * hidden_size) \
            + seq_len * hidden_size * 2
    else:
        mem_io = batch_size * (seq_len * hidden_size) \
            + 1 * hidden_size * 2
    return mem_io , output_shape

def ReplicatedLinear(input_dim, output_dim, input_shape, decode = False):
    batch_size, seq_len, hidden_size = input_shape
    assert input_dim == hidden_size
    output_shape = (batch_size, seq_len, output_dim)
    mem_io = batch_size * seq_len * input_dim \
            + batch_size * seq_len * output_dim \
            + input_dim * output_dim
    # if decode:
    #     #weight read
    #     mem_io +=  input_dim * output_dim * (seq_len - 1)
    return mem_io, output_shape

def ParallelLinear(input_dim, output_dim, input_shape, mode = 0, decode =False, moe=False):
    global TP_SIZE
    batch_size, seq_len, hidden_size = input_shape
    comm_io=0
    if mode == 0:
        assert input_dim == hidden_size
        mem_io = batch_size * seq_len * input_dim \
                + batch_size * seq_len * output_dim/TP_SIZE
        if not moe:
            if decode:
                mem_io += input_dim * output_dim/TP_SIZE * seq_len
            else:
                mem_io += input_dim * output_dim/TP_SIZE
        else:
            if decode:
                mem_io += input_dim * output_dim/TP_SIZE * min(batch_size, 32) * seq_len
            else:
                mem_io += input_dim * output_dim/TP_SIZE * min(batch_size * seq_len, 32)


        output_shape = (batch_size, seq_len, output_dim/TP_SIZE)
    else:
        assert input_dim/TP_SIZE  == hidden_size
        mem_io = batch_size * seq_len * input_dim/TP_SIZE \
                + batch_size * seq_len * output_dim
        if not moe:
            if decode:
                mem_io += input_dim * output_dim/TP_SIZE * seq_len
            else:
                mem_io += input_dim * output_dim/TP_SIZE
        else:
            if decode:
                mem_io += input_dim * output_dim/TP_SIZE * min(batch_size, 32) * seq_len
            else:
                mem_io += input_dim * output_dim/TP_SIZE * min(batch_size * seq_len, 32)
            #weight read

        if TP_SIZE > 1:
            # reduce
            mem_io+= batch_size * seq_len * output_dim * 2 * (TP_SIZE - 1)/TP_SIZE
            comm_io = batch_size * seq_len * output_dim * 2 * (TP_SIZE - 1)/TP_SIZE
        output_shape = (batch_size, seq_len, output_dim)
    return mem_io, output_shape, comm_io

def VocabParallelEmbedding(vocab_size, embedding_dim, input_shape):
    batch_size, seq_len = input_shape
    output_shape = (batch_size, seq_len, embedding_dim)
    mem_io = batch_size * (seq_len + seq_len * hidden_size)
    return mem_io , output_shape

def DeepSeekV2MLP(hidden_size, intermediate_size, input_shape, decode = False, moe=False):
    global TP_SIZE
    batch_size , seq_len, hidden_dim = input_shape
    assert hidden_dim == hidden_size
    mem_io = 0
    mem_io0, out0_shape, comm_io0 = ParallelLinear(hidden_size, intermediate_size * 2, input_shape, mode=0, decode = decode, moe=moe)

    #silu and mul
    _, _, out0_dim = out0_shape
    assert out0_dim == intermediate_size/TP_SIZE * 2
    mem_io1 = batch_size * seq_len * intermediate_size/TP_SIZE * 3

    mem_io2, out1_shape, comm_io1 = ParallelLinear(intermediate_size, hidden_size, (batch_size, seq_len, intermediate_size/TP_SIZE), mode=1, decode = decode, moe=moe)
    mem_io = mem_io0 + mem_io1 + mem_io2
    _, _, out1_dim = out1_shape
    assert out1_dim == hidden_size

    return mem_io, out1_shape, comm_io0 + comm_io1


def DeepSeekV2Moe(hidden_size,intermediate_size, n_routed_experts, num_experts, input_shape, decode = False):
    global TP_SIZE
    num_experts = 8
    batch_size , seq_len, hidden_size = input_shape
    mem_io = 0
    #gate
    mem_io0 = batch_size * (seq_len * hidden_size + seq_len * n_routed_experts) \
            + hidden_size * n_routed_experts

    #shared
    mem_io1, shared_output_shape, comm_io0  = DeepSeekV2MLP(hidden_size, intermediate_size, input_shape, decode = decode)

    # experts
    mem_io2, experts_output_shape, comm_io1 = DeepSeekV2MLP(hidden_size, moe_intermediate_size * num_experts, input_shape, decode = decode, moe=True)


    #sum
    mem_io3 =  batch_size * seq_len * hidden_size * 2

    assert experts_output_shape == input_shape
    mem_io = mem_io0 + mem_io1 +mem_io2 + mem_io3
    return mem_io, input_shape, comm_io0 + comm_io1

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
                           decode=False,
                           prefill= 0):
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

    mem_io = 0
    #q_a_proj
    mem_io0, ckq_shape = ReplicatedLinear(hidden_size, q_lora_rank , input_shape, decode = decode)

    #q_a_layernorm
    mem_io1, hidden_states_or_q_c_shape = RMSNorm(q_lora_rank, ckq_shape)

    #kv_a_proj_with_mqa
    mem_io2, k_cpe_shape = ReplicatedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim, input_shape, decode = decode)
    B, S, k_cpe_dim = k_cpe_shape
    assert k_cpe_dim == kv_lora_rank + qk_rope_head_dim
    kv_c_shape = (B, S, kv_lora_rank)
    k_pe_shape = (B, S, qk_rope_head_dim)

    #kv_a_layernorm
    mem_io3, kv_c_normed_shape = RMSNorm(kv_lora_rank, kv_c_shape)

    #self.mla_attn(hidden_states_or_q_c, kv_c_normed, k_pe)
    # hidden_states_or_q_c = b, s, q_lora_rank
    # kv_c_normed_shape = kv_c_shape = (B, S, kv_lora_rank)
    #k_pe_shape = (B, S, qk_rope_head_dim)

    # q,k, v projection
    mem_io4, q_b_proj_output_shape, comm_io0 = ParallelLinear(q_lora_rank, num_heads * qk_head_dim, hidden_states_or_q_c_shape, mode=0, decode=decode)

    mem_io5, kb_b_proj_output_shape, comm_io1 = ParallelLinear(kv_lora_rank,
                                                        num_heads * (qk_nope_head_dim + v_head_dim),
                                                        kv_c_normed_shape,
                                                        mode=0,
                                                        decode=decode)


    ## Attention
    assert num_heads * qk_head_dim / TP_SIZE == q_b_proj_output_shape[2]
    assert num_heads * (qk_nope_head_dim + v_head_dim) / TP_SIZE == kb_b_proj_output_shape[2]

    mem_io6 = batch_size * seq_len * num_heads * qk_head_dim/TP_SIZE * 2 \
            + batch_size * seq_len * num_heads * v_head_dim/TP_SIZE  \
            + batch_size * seq_len * num_heads * v_head_dim/TP_SIZE
    kv_cache_io = 0
    if decode:
        kv_cache_io = batch_size * kv_lora_rank * seq_len * (2 * prefill + seq_len - 1) / 2
        # print("KV Cache(GB):", kv_cache_io * FLOAT16 / GB)
        mem_io6 += kv_cache_io


    attention_output_shape  = (batch_size, seq_len, v_head_dim * num_heads/TP_SIZE)
    mem_io7, o_proj_output_shape, comm_io2 = ParallelLinear(v_head_dim * num_heads, hidden_size,  attention_output_shape, mode=1, decode=decode)

    mem_io = mem_io0 + mem_io1+ mem_io2 + mem_io3 + mem_io4 + mem_io5 + mem_io6 + mem_io7


    assert o_proj_output_shape == (batch_size, seq_len, hidden_size)
    #q_proj = (B, Seq, 192)
    #k prrj - (B, Seq, 192)
    #v proj = (B, Seq, 128)

    return mem_io, o_proj_output_shape, kv_cache_io, comm_io0+comm_io1+comm_io2

def LMHead(hidden_size, vocab_size, input_shape, decode = False):
    global TP_SIZE
    batch_size, seq_len, hidden_dim = input_shape
    assert hidden_size == hidden_dim

    mem_io = seq_len * hidden_size * batch_size \
            +hidden_size * vocab_size/TP_SIZE \
            +batch_size * seq_len * vocab_size/TP_SIZE

    if decode:
        mem_io += hidden_size * vocab_size/TP_SIZE * (seq_len-1)

    out_shape = (batch_size, seq_len, vocab_size/TP_SIZE)
    return mem_io, out_shape



if __name__ == "__main__":
    # config
    import argparse
    parser = argparse.ArgumentParser(description="Process batch, prefill length, and decode length.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--prefill-len", type=int, default=1, help="Prefill length")
    parser.add_argument("--decode-len", type=int, default=2, help="Decode length")

    args = parser.parse_args()

    print(f"BATCH = {args.batch}")
    print(f"PREFILL_LEN = {args.prefill_len}")
    print(f"DECODE_LEN = {args.decode_len}")


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

    prefill_mem_io = 0
    decode_mem_io = 0
    total_kv_cache_io = 0


# PREFILL
    batch_size = BATCH
    seq_len = PREFILL_LEN
    input = (batch_size, seq_len)
    prefill_mem_io, embeded_tokens = VocabParallelEmbedding(vocab_size, hidden_size, input)
    prefill_comm_io = 0

    for i in range(3):
        prefill_mem_io0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
        prefill_mem_io += prefill_mem_io0
        prefill_mem_io1, attn_out, _, comm_io0 = DeepseekV2MLAAttention(hidden_size,
                            num_attention_heads,
                            qk_nope_head_dim,
                            qk_rope_head_dim,
                            v_head_dim,
                            q_lora_rank,
                            kv_lora_rank,
                            max_position_embedding,
                            input_layernorm)
        prefill_mem_io += prefill_mem_io1
        prefill_mem_io2, output_layernorm = RMSNorm(hidden_size, attn_out)
        prefill_mem_io += prefill_mem_io2
        prefill_mem_io3, mlp_out, comm_io1 = DeepSeekV2MLP(hidden_size, intermediate_size, output_layernorm)
        prefill_mem_io += prefill_mem_io3
        prefill_comm_io += comm_io0
        prefill_comm_io += comm_io1

    for i in range(num_hidden_layers - 3):
        prefill_mem_io0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
        prefill_mem_io += prefill_mem_io0
        prefill_mem_io1, attn_out, _, comm_io2 = DeepseekV2MLAAttention(hidden_size,
                            num_attention_heads,
                            qk_nope_head_dim,
                            qk_rope_head_dim,
                            v_head_dim,
                            q_lora_rank,
                            kv_lora_rank,
                            max_position_embedding,
                            input_layernorm)
        prefill_mem_io += prefill_mem_io1
        prefill_mem_io2, output_layernorm = RMSNorm(hidden_size, attn_out)
        prefill_mem_io += prefill_mem_io2
        prefill_mem_io3, mlp_out, comm_io3 = DeepSeekV2Moe(hidden_size, intermediate_size,  n_routed_experts, num_experts_per_token, output_layernorm)
        prefill_mem_io += prefill_mem_io3
        prefill_comm_io += comm_io2
        prefill_comm_io += comm_io3


    prefill_mem_io_norm, output_layernorm = RMSNorm(hidden_size, mlp_out)
    prefill_mem_io += prefill_mem_io_norm

    prefill_mem_io_head, output_head = LMHead(hidden_size, vocab_size, mlp_out)
    prefill_mem_io += prefill_mem_io_head


    if DECODE_LEN > 1:
        batch_size = BATCH
        seq_len = DECODE_LEN - 1
        input = (batch_size, seq_len)

        decode_mem_io, embeded_tokens = VocabParallelEmbedding(vocab_size, hidden_size, input)

        total_kv_cache_io = 0
        attention_io = 0
        decode_comm_io = 0
        for i in range(3):
            decode_mem_io0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
            decode_mem_io += decode_mem_io0
            decode_mem_io1, attn_out, kv_cache_io, comm_io0 = DeepseekV2MLAAttention(hidden_size,
                                num_attention_heads,
                                qk_nope_head_dim,
                                qk_rope_head_dim,
                                v_head_dim,
                                q_lora_rank,
                                kv_lora_rank,
                                max_position_embedding,
                                input_layernorm,
                                decode = True,
                                prefill= PREFILL_LEN)
            attention_io += decode_mem_io1
            decode_mem_io += decode_mem_io1
            decode_mem_io2, output_layernorm = RMSNorm(hidden_size, attn_out)
            decode_mem_io += decode_mem_io2
            decode_mem_io3, mlp_out, comm_io1 = DeepSeekV2MLP(hidden_size, intermediate_size, output_layernorm, decode = True)
            decode_mem_io += decode_mem_io3
            total_kv_cache_io += kv_cache_io
            decode_comm_io += comm_io0
            decode_comm_io += comm_io1

        for i in range(num_hidden_layers - 3):
            decode_mem_io0, input_layernorm  = RMSNorm(hidden_size, embeded_tokens)
            decode_mem_io += decode_mem_io0
            decode_mem_io1, attn_out,kv_cache_io, comm_io2 = DeepseekV2MLAAttention(hidden_size,
                                num_attention_heads,
                                qk_nope_head_dim,
                                qk_rope_head_dim,
                                v_head_dim,
                                q_lora_rank,
                                kv_lora_rank,
                                max_position_embedding,
                                input_layernorm,
                                decode = True,
                                prefill= PREFILL_LEN)
            attention_io += decode_mem_io1
            decode_mem_io += decode_mem_io1
            decode_mem_io2, output_layernorm = RMSNorm(hidden_size, attn_out)
            decode_mem_io += decode_mem_io2
            decode_mem_io3, mlp_out, comm_io3 = DeepSeekV2Moe(hidden_size, intermediate_size,  n_routed_experts, num_experts_per_token, output_layernorm, decode= True)
            decode_mem_io += decode_mem_io3
            total_kv_cache_io += kv_cache_io
            decode_comm_io += comm_io2
            decode_comm_io += comm_io3


        decode_mem_io_norm, output_layernorm = RMSNorm(hidden_size, mlp_out)
        decode_mem_io += decode_mem_io_norm

        decode_mem_io_head, output_head = LMHead(hidden_size, vocab_size, mlp_out, decode= True)
        decode_mem_io += decode_mem_io_head
        # print("ATTENTION_IO: ", attention_io/GB)

    print("PREFILL: ", prefill_mem_io/GB)
    print("DECODE: ", decode_mem_io/GB)
    print("KV_CACHE_IO: ", total_kv_cache_io/GB)
    print("PREFILL COMM: ", prefill_comm_io/GB)
    print("DECODE COMM: ", decode_comm_io/GB)
    print("TOTAL_MEM_REQUIRED: ", (prefill_mem_io + decode_mem_io + total_kv_cache_io * FLOAT16)/GB)
