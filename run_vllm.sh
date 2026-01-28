uv run vllm serve arcee-ai/trinity-mini \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --generation-config vllm \
    --speculative_config '{"model": "./work/checkpoints/draft-model.eagle3/49", "num_speculative_tokens": 2, "method": "eagle3"}'
