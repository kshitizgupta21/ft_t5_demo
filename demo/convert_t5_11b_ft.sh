python3 FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir ft_t5-11b \
        -in_file t5-11b \
        -inference_tensor_para_size 4 \
        -weight_data_type fp32
