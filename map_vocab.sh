uv run scripts/build_vocab_mapping.py \
    --token-freq-path ./work/token_freq.pt \
    --draft-vocab-size 32000 \
    --target-model-path arcee-ai/trinity-mini \
    --output-path ./work