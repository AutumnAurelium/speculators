uv run python scripts/data_generation_offline.py \
    --target-model-path arcee-ai/trinity-mini \
    --train-data-path work/int3_converted.jsonl \
    --output-dir ./work/int3_processed \
    --seq-length 2048 \
    --batch-size 8 \
    --max-samples 50000 \
    --gpu-memory-utilization 0.9 \
    --layer-ids 1 15 28 31