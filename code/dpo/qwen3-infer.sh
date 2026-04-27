export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /path/to/models/qwen3-32b-merged \
    --served-model-name qwen3-32b-merged \
    --host 0.0.0.0 \
    --port 48000 \
    --gpu-memory-utilization 0.9 \
    --data-parallel-size=1 \
    --tensor-parallel-size=4 \
    --max-model-len 16384