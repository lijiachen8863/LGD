CUDA_VISIBLE_DEVICES=0 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0.2
CUDA_VISIBLE_DEVICES=0 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0.2
CUDA_VISIBLE_DEVICES=0 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0.2
