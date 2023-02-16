# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

import argparse
import numpy as np
import os
import sys
import torch
import torch.distributed as dist


from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from FasterTransformer.examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from FasterTransformer.examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str, required=True)
    parser.add_argument('--inference_data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument('--lib_path', type=str, default='./FasterTransformer/build/lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('-len_penalty', '--len_penalty', type=float, default=0.0, metavar='NUMBER',
                        help='Length penalty for generating tokens. Default is 0.0.')
    parser.add_argument('--diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='diversity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beam search.')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, metavar='NUMBER',
                        help='Repetition penalty for generating tokens. Default is 1.0.')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='NUMBER',
                        help='Temperature penalty for generating tokens. Default is 1.0.')
    parser.add_argument("--topk", type=int, default=1, help="top k for sampling")
    parser.add_argument("--topp", type=float, default=0.0, help="top p for sampling")
    parser.add_argument("--beam_width", type=int, default=1, help="beam width for beam search")

    args = parser.parse_args()

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0


    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"

    
    # Define Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_lenth=128)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length=512

    # Define Inputs
    INPUTS = [
    "translate English to French: Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems",
    "translate English to German: Giant sequoia trees are the largest trees by volume in the world"
    ]
    batch_size = len(INPUTS)
    inputs = tokenizer(INPUTS, padding=True, return_tensors="pt")


    ckpt_config = configparser.ConfigParser()

    ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    else:
        assert False, "[ERROR] This example only support loading model with FT format directly."

    weight_data_type = np.float32
    weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
    relative_attention_max_distance = 128
    encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                d_model=ckpt_config.getint("encoder", "d_model"),
                                d_kv=ckpt_config.getint("encoder", "d_kv"),
                                d_ff=ckpt_config.getint("encoder", "d_ff"),
                                num_layers=ckpt_config.getint("encoder", "num_layers"),
                                num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                num_heads=ckpt_config.getint("encoder", "num_heads"),
                                relative_attention_num_buckets=ckpt_config.getint(
                                    "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                )
    decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                d_model=ckpt_config.getint("decoder", "d_model"),
                                d_kv=ckpt_config.getint("decoder", "d_kv"),
                                d_ff=ckpt_config.getint("decoder", "d_ff"),
                                num_layers=ckpt_config.getint("decoder", "num_layers"),
                                num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                num_heads=ckpt_config.getint("decoder", "num_heads"),
                                relative_attention_num_buckets=ckpt_config.getint(
                                    "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                )

    t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
    use_gated_activation = encoder_config.is_gated_act
    position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
    activation_type = encoder_config.feed_forward_proj


    tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    ft_encoder_weight.load_from_bin(ft_model_location, "Megatron")


    ft_decoding_weight.load_from_bin(ft_model_location, "Megatron")

    if args.inference_data_type == "fp32":
        ft_encoder_weight.to_float()
        ft_decoding_weight.to_float()
    elif args.inference_data_type == "fp16":
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()
    elif args.inference_data_type == "bf16":
        ft_encoder_weight.to_bfloat16()
        ft_decoding_weight.to_bfloat16()

    ft_encoder_weight.to_cuda()
    ft_decoding_weight.to_cuda()

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    remove_padding = True if batch_size > 32 else False
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets,
                                0, # num_experts
                                [], # moe_layer_index
                                relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                pipeline_para_size, t5_with_bias,
                                position_embedding_type, moe_k=0, activation_type=activation_type)

    ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                decoder_config.vocab_size, q_scaling,
                                decoder_config.relative_attention_num_buckets,
                                0, # num_experts
                                [], # moe_layer_index
                                max_distance=relative_attention_max_distance,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                moe_k=0, activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

    ft_t5 = FTT5(ft_encoder, ft_decoding)



    with torch.no_grad():
        ft_output_ids, ft_sequence_length = ft_t5(input_token=inputs,
                                                        inputs_embeds=None,
                                                        beam_size=args.beam_width,
                                                        max_seq_len=args.max_seq_len,
                                                        top_k=args.topk,
                                                        top_p=args.topp,
                                                        beam_search_diversity_rate=args.diversity_rate,
                                                        is_return_output_log_probs=False,
                                                        is_return_cum_log_probs=False,
                                                        repetition_penalty=args.repeat_penalty,
                                                        temperature=args.temperature,
                                                        len_penalty=args.len_penalty,
                                                        bad_words_list=None,
                                                        stop_words_list=None)
    ft_outputs = []
    for i in range(batch_size):
        # selecting the top sequence from beam width number of sequences
        ft_outputs.append(list(ft_output_ids[i, 0, :][:ft_sequence_length[i , 0]]))
    ft_tokens = tokenizer.batch_decode(ft_outputs, skip_special_tokens=True)
    if rank == 0:
        print("\n FT Output:")
        print(ft_tokens)

if __name__ == '__main__':
    main()
