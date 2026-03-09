
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco/refcoco_val.yaml
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco/refcoco_testA.yaml
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco/refcoco_testB.yaml