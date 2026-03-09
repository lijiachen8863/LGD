CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_val.yaml --llm-model LLaVA_qwen_0.5b
CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_testA.yaml --llm-model LLaVA_qwen_0.5b
CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_testB.yaml --llm-model LLaVA_qwen_0.5b

CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_val.yaml --llm-model LLaVA_qwen_7b
CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_testA.yaml --llm-model LLaVA_qwen_7b
CUDA_VISIBLE_DEVICES=1 python llm_tas_main_qw.py --config config/refcoco/refcoco_testB.yaml --llm-model LLaVA_qwen_7b

