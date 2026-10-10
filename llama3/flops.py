import yaml
import argparse
import math

tflops = 2**40
# YAML 파일을 읽어 설정 값을 가져오는 함수
def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prefill(seq_len, tp_size, config):
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']




    input_layernorm = seq_len * hidden_size * 8
    qkv_linear = (2 * seq_len * hidden_size  * hidden_size / tp_size) + \
            (2 * seq_len * hidden_size *  (hidden_size/tp_size)/(num_attention_heads/num_key_value_heads) * 2)

    self_attention = seq_len * seq_len * (hidden_size/tp_size) * 2 \
            + seq_len * (hidden_size/tp_size) * seq_len * 2 \
            + seq_len * seq_len * 7

    out_linear = 2 * seq_len * (hidden_size/tp_size) * hidden_size
    if tp_size > 1:
        out_linear += math.log2(tp_size) * seq_len * hidden_size

    attention = qkv_linear + self_attention + out_linear

    residual0 =  seq_len * hidden_size

    post_layernorm = seq_len * hidden_size * 8

    feedforward = 2 * seq_len * hidden_size * (intermediate_size/tp_size) \
            + 2 * seq_len * hidden_size * (intermediate_size/tp_size) \
            + 9 * seq_len * (intermediate_size/tp_size) \
            + 2 * seq_len * hidden_size * (intermediate_size/tp_size)
    if tp_size > 1:
        feedforward += math.log2(tp_size) * seq_len * hidden_size

    residual1 = seq_len * hidden_size

    decoding_layer = num_hidden_layers * (input_layernorm + attention + residual0 + post_layernorm + feedforward + residual1)

    last_layernorm = seq_len * hidden_size * 8
    lm_head = (2 * seq_len * hidden_size * vocab_size)/tp_size
    not_decoding_layer = last_layernorm + lm_head

    total = not_decoding_layer + decoding_layer
    total_tflops = total / tflops
    print("decoding: prefill - ", decoding_layer)
    print("nondecoding: prefill - ", not_decoding_layer)

    return total_tflops

def decode(prefill_len, decode_len, tp_size, config):
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']


    input_layernorm = decode_len * hidden_size * 8
    qkv_linear = (2 * decode_len * hidden_size * hidden_size/tp_size)  \
            + (2 * decode_len * hidden_size *  (hidden_size/tp_size)/(num_attention_heads/num_key_value_heads)) * 2

    self_attention = decode_len * (2 * prefill_len + decode_len + 1 ) * (hidden_size/tp_size) \
            + decode_len * (2 * prefill_len + decode_len + 1 ) * (hidden_size/tp_size) \
            + decode_len *  decode_len/2 * (2 * prefill_len + decode_len + 1) * 7

    out_linear = 2 * decode_len * (hidden_size/tp_size) * hidden_size
    if tp_size > 1:
        out_linear += math.log2(tp_size) * decode_len * hidden_size

    attention = qkv_linear + self_attention + out_linear

    residual0 =  decode_len * hidden_size
    post_layernorm = decode_len * hidden_size * 8

    feedforward = 2 * decode_len * hidden_size * (intermediate_size/tp_size) \
            + 2 * decode_len * hidden_size * (intermediate_size/tp_size) \
            + 9 * decode_len * (intermediate_size/tp_size) \
            + 2 * decode_len * hidden_size * (intermediate_size/tp_size)
    if tp_size > 1:
        feedforward += math.log2(tp_size) * decode_len * hidden_size

    residual1 = decode_len * hidden_size

    decoding_layer = num_hidden_layers * (input_layernorm + attention + residual0 + post_layernorm + feedforward + residual1)

    last_layernorm = decode_len * hidden_size * 8
    lm_head = (2 * decode_len * hidden_size * vocab_size)/tp_size
    not_decoding_layer = last_layernorm + lm_head

    total = not_decoding_layer + decoding_layer
    total_tflops = total / tflops
    # print(attention / tflops * 1000)
    # print(feedforward /tflops * 1000)
    print("decoding: decoode - ", decoding_layer)
    print("nondecoding: decode - ", not_decoding_layer)
    return total_tflops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate TFLOPs for prefill and decode stages.")
    parser.add_argument("--config", type=str, default="./llama3_8b.yaml", help="Path to the config YAML file")
    parser.add_argument("--input_len", type=int, default=128, help="Length of the input sequence")
    parser.add_argument("--output_len", type=int, default=128, help="Length of the output sequence")
    parser.add_argument("--batch_size", type=float, default=1., help="Batch size")
    parser.add_argument("--tp_size", type=int, default=1, help="TP size")

    args = parser.parse_args()
    config = load_config(args.config)

    prefill_tflops = prefill(args.input_len, args.tp_size, config)
    decode_tflops = decode(args.input_len, args.output_len-1, args.tp_size, config)
    print("==============================================")
    print(f"TP: {args.tp_size}, input: {args.input_len}, output:{args.output_len}")
    print("Total TFlops: ", args.batch_size * (prefill_tflops + decode_tflops))
    print("Prefill TFlops: ", args.batch_size * (prefill_tflops))
    print("Decode TFlops: ", args.batch_size * (decode_tflops))
    print("==============================================")
