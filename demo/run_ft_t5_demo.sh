mpirun -n 4 --allow-run-as-root python3 ft_t5_demo.py \
                    --ft_model_location ft_t5-11b/ \
                    --inference_data_type bf16 \
                    --tensor_para_size 4
