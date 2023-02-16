#!/usr/bin/env python3
import os
import argparse
import ast
import json

parser = argparse.ArgumentParser()
parser.add_argument("--hf-model-config-path", required=True)
parser.add_argument("--data-type", type=int, help="Inference Data Type, Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)", default=1)
parser.add_argument("--batch-size", type=int, help="batch size", default=1)
parser.add_argument("--max-input-len", type=int, help="max input seq len to tune for", default=64)
parser.add_argument("--beam-width", type=int, help="beam width", default=1)
parser.add_argument("--tensor-para-size", type=int, help="Tensor Parallelism Degree aka Num of GPUs for inference", default=1)
parser.add_argument("--is-append", type=int, help="whether to append the config generated in this run to existing gemm config file, default False i.e. 0", default=0)

# example usage: python3 gen_t5gemm_config.py --hf-model-config-path t5-11b/config.json --data-type 2 --tensor-para-size 4

args = parser.parse_args()
max_input_len = args.max_input_len
beam_width = args.beam_width
batch_size = args.batch_size

with open(args.hf_model_config_path) as config_file:
    parsed_config = json.load(config_file)


d_model = parsed_config["d_model"]
head_num = parsed_config["num_heads"]
inter_size = parsed_config["d_ff"]
# inter_size = 4 * head_number * size_per_head
# size_per_head = inter_size / (4 * head_number)
size_per_head = inter_size / (4 * head_num)
decoder_vocab_size = parsed_config["vocab_size"]

encoder_d_model = decoder_d_model = d_model
encoder_head_num = decoder_head_num  = head_num 
encoder_size_per_head = decoder_size_per_head = size_per_head
encoder_inter_size = decoder_inter_size = inter_size

data_type = args.data_type
tensor_para_size = args.tensor_para_size
is_append = args.is_append 

os.system(f"./FasterTransformer/build/bin/t5_gemm {batch_size} {beam_width} {max_input_len} {encoder_d_model} {encoder_head_num} {encoder_size_per_head} {encoder_inter_size} {decoder_d_model} {decoder_head_num} {decoder_size_per_head} {decoder_inter_size} {decoder_vocab_size} {data_type} {tensor_para_size} {is_append}")

'''
Settings of ./bin/t5_gemm are 

batch_size
beam_width 
max_mem_seq_len
encoder_d_model 
encoder_head_num 
encoder_size_per_head
encoder_inter_size 
decoder_d_model 
decoder_head_num 
decoder_size_per_head 
decoder_inter_size 
decoder_vocab_size 
data_type = 0 (FP32) or 1 (FP16) or 2 (BF16)
tensor_para_size 
'''

