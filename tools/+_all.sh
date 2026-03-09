
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco+/refcoco_p_val.yaml
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco+/refcoco_p_testA.yaml
CUDA_VISIBLE_DEVICES=1 python llm_tas_main.py --config config/refcoco+/refcoco_p_testB.yaml