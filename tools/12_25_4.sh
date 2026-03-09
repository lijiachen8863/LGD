CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0.4
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0.4
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0.4

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0.3
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0.3
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0.3

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0.1
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0.1
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0.1

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_val.yaml  --Weight_S_l 0.9
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testA.yaml  --Weight_S_l 0.9
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco/refcoco_testB.yaml  --Weight_S_l 0.9

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco+/refcoco_p_val.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco+/refcoco_p_testA.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcoco+/refcoco_p_testB.yaml  --Weight_S_l 0

CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcocog/refcocog_g.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcocog/refcocog_u_test.yaml  --Weight_S_l 0
CUDA_VISIBLE_DEVICES=1 python calculate_multiple_config.py --config config/refcocog/refcocog_u_val.yaml  --Weight_S_l 0
