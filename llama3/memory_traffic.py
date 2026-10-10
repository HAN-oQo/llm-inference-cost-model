import yaml
import argparse
import math

GB = 2**30
TB = 2**40
FLOAT16 = 2

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prefill(batch_size, seq_len, tp_size, config):
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']
    max_position_embeddings = config['max_position_embeddings']

    embedding_io = batch_size * (seq_len + seq_len * hidden_size)

    #Decoding Layer

    ## LayerNorm
    input_layernorm_io = batch_size * (seq_len * hidden_size * 2)
    input_layernorm_weight = 1 * hidden_size * 2

    ## Attention
    qkv_linear_io = batch_size * (seq_len * hidden_size \
                + seq_len * hidden_size * 2 \
                + seq_len * (hidden_size/tp_size) \
                + seq_len * (hidden_size/tp_size) /(num_attention_heads  / num_key_value_heads ) * 2)
    qkv_linear_weight = hidden_size * (hidden_size/tp_size) \
                + hidden_size  *  (hidden_size/tp_size) /(num_attention_heads  / num_key_value_heads ) * 2 \

    self_attention_io = batch_size * (seq_len * (hidden_size/tp_size)  + seq_len * (hidden_size/tp_size) /(num_attention_heads  / num_key_value_heads ) * 2 \
                + seq_len * (hidden_size/tp_size))

    output_linear_io = batch_size * (seq_len * (hidden_size/tp_size) \
                + seq_len * hidden_size)
    if tp_size > 1:
        output_linear_io += batch_size * seq_len * hidden_size * (tp_size)

    output_linear_weight = (hidden_size/tp_size) * hidden_size

    ## Residual
    residual1_io = batch_size * (seq_len * hidden_size * 4)
    ## & LayerNorm
    output_layernorm_io = batch_size * (seq_len * hidden_size * 2)
    output_layernorm_weight = 1 * hidden_size * 2

    ## FeedForward
    feedforward_io = batch_size * (seq_len * hidden_size * 2\
                + seq_len * (intermediate_size / tp_size) * 2\
                + seq_len * (intermediate_size / tp_size) * 2 \
                + seq_len * (intermediate_size / tp_size) \
                + seq_len * (intermediate_size / tp_size) \
                + seq_len * (intermediate_size / tp_size) \
                # + seq_len * intermediate_size \
                + seq_len * hidden_size)
    if tp_size > 1:
        feedforward_io += batch_size * seq_len * hidden_size * (tp_size)
    feedforward_weight = hidden_size * (intermediate_size / tp_size) * 2\
                + hidden_size * intermediate_size / tp_size
    ## Residual
    residual2_io  = batch_size * (seq_len * hidden_size * 4)

    # LayerNorm
    last_layernorm_io = batch_size * (seq_len * hidden_size * 2)
    last_layernorm_weight = 1 * hidden_size * 2

    # LM Head
    lm_head_io = batch_size * (seq_len * hidden_size + seq_len * vocab_size/tp_size)
    lm_head_weight =  hidden_size * vocab_size/tp_size


    attention_io = (self_attention_io) * num_hidden_layers
    attention_weight = 0
    non_attention_io = embedding_io + (qkv_linear_io + output_linear_io + input_layernorm_io + residual1_io + output_layernorm_io + feedforward_io + residual2_io) * num_hidden_layers + last_layernorm_io + lm_head_io
    non_attention_weight = (input_layernorm_weight + qkv_linear_weight + output_linear_weight + output_layernorm_weight + feedforward_weight)  * num_hidden_layers + last_layernorm_weight + lm_head_weight
    total  = attention_io + attention_weight + non_attention_io + non_attention_weight

    print("Prefill Logs")
    print("attention io: ", attention_io * FLOAT16 / GB)
    print("non_attention io: ", non_attention_io * FLOAT16 / GB)
    print("non_attention_weight: ", non_attention_weight * FLOAT16 / GB)

    return attention_io, attention_weight, non_attention_io, non_attention_weight, total


# layernome 때문에 weight read 가 딱 N배는 아님. 조금 작음.
def decode(batch_size, prefill_len, decode_len, tp_size, config):
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']
    max_position_embeddings = config['max_position_embeddings']

    embedding_io = batch_size * (decode_len + decode_len * hidden_size)

    input_layernorm_io = batch_size * (decode_len * hidden_size * 2)
    input_layernorm_weight = decode_len * hidden_size * 2


    qkv_linear_io = batch_size * (decode_len * hidden_size \
                + decode_len * hidden_size * 2 \
                + decode_len * (hidden_size/tp_size) \
                + decode_len * (hidden_size/tp_size) /(num_attention_heads  / num_key_value_heads ) * 2)

    qkv_linear_weight = decode_len * hidden_size * (hidden_size/tp_size) \
                + decode_len * hidden_size * (hidden_size/tp_size)/(num_attention_heads  / num_key_value_heads ) * 2 \

    self_attention_io = batch_size * (decode_len *(hidden_size/tp_size)  + decode_len * (hidden_size/tp_size)/(num_attention_heads  / num_key_value_heads ) * 2 \
                + decode_len * (hidden_size/tp_size))

    kvcache_io = batch_size * (decode_len *  (2*prefill_len + decode_len -1) \
            * (hidden_size/tp_size) /(num_attention_heads/num_key_value_heads))


    output_linear_io = batch_size * (decode_len * (hidden_size/tp_size) + decode_len * hidden_size)
    if tp_size > 1:
        output_linear_io += batch_size * decode_len * hidden_size * (tp_size)
    output_linear_weight =  decode_len * (hidden_size/tp_size) * hidden_size


    residual1_io = batch_size * (decode_len * hidden_size*4)


    output_layernorm_io = batch_size * (decode_len * hidden_size * 2)
    output_layernorm_weight = decode_len * hidden_size * 2


    ## FeedForward
    feedforward_io = batch_size * (decode_len * hidden_size * 2\
                + decode_len * (intermediate_size/tp_size) * 2\
                + decode_len * (intermediate_size/tp_size) * 2 \
                + decode_len * (intermediate_size/tp_size) \
                + decode_len * (intermediate_size/tp_size) \
                + decode_len * (intermediate_size/tp_size) \
                # + decode_len * intermediate_size \
                + decode_len * hidden_size )
    if tp_size > 1:
        feedforward_io += batch_size * decode_len * hidden_size * (tp_size)

    feedforward_weight = decode_len * hidden_size * intermediate_size/tp_size * 2\
                + decode_len * hidden_size * intermediate_size/tp_size

    residual2_io  = batch_size * (decode_len * hidden_size * 4)

    # LayerNorm
    last_layernorm_io = batch_size * (decode_len * hidden_size * 2)
    last_layernorm_weight = decode_len * hidden_size * 2

    lm_head_io =batch_size * (decode_len * hidden_size + decode_len * vocab_size/tp_size)
    lm_head_weight = decode_len * hidden_size * vocab_size/tp_size

    attention_io = (self_attention_io) * num_hidden_layers
    attention_weight = 0
    non_attention_io = embedding_io + (qkv_linear_io + output_linear_io + input_layernorm_io + residual1_io + output_layernorm_io + feedforward_io + residual2_io) * num_hidden_layers + last_layernorm_io + lm_head_io
    non_attention_weight = (input_layernorm_weight + qkv_linear_weight + output_linear_weight + output_layernorm_weight + feedforward_weight)  * num_hidden_layers + last_layernorm_weight + lm_head_weight
    kvcache_io = kvcache_io * num_hidden_layers
    total  = kvcache_io + attention_io + attention_weight + non_attention_io + non_attention_weight
    print("Decode Logs")
    print("attention io: ", attention_io * FLOAT16 / GB)
    print("non_attention io: ", non_attention_io * FLOAT16 / GB)
    print("non_attention_weight: ", non_attention_weight * FLOAT16 / GB)
    print("KV cache io: ", kvcache_io * FLOAT16 / GB)
    return attention_io, attention_weight, non_attention_io, non_attention_weight, kvcache_io, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MEMORY Traffic for prefill and decode stages.")
    parser.add_argument("--config", type=str, default="./llama3_8b.yaml", help="Path to the config YAML file")
    parser.add_argument("--input_len", type=int, default=128, help="Length of the input sequence")
    parser.add_argument("--output_len", type=int, default=128, help="Length of the output sequence")
    parser.add_argument("--batch_size", type=float, default=1., help="Batch size")
    parser.add_argument("--tp_size", type=int, default=1, help="TP size")

    args = parser.parse_args()
    config = load_config(args.config)


    prefill_attn_io, prefill_attn_weight, prefill_nonattn_io, prefill_nonattn_weight, prefill_total = prefill(args.batch_size, args.input_len, args.tp_size, config)
    decode_attn_io, decode_attn_weight, decode_nonattn_io, decode_nonattn_weight, kvcache,  decode_total = decode(args.batch_size, args.input_len, args.output_len-1, args.tp_size, config)

    print("==============================================")
    print(f"TP: {args.tp_size}, input: {args.input_len}, output:{args.output_len}")
    print("prefill: ", prefill_total * FLOAT16 / GB)
    print("decode: ", decode_total * FLOAT16 / GB)
    print("total: ", (prefill_total + decode_total) * FLOAT16 / GB)
    print("==============================================")
