#!/usr/bin/env python3
"""Estimate parameter count for Eagle3 draft model."""

import argparse

import numpy as np
from transformers import AutoConfig


def count_params(
    verifier_name: str,
    num_layers: int,
    draft_vocab_size: int | None = None,
    d2t_path: str | None = None,
):
    """Count trainable and total parameters for Eagle3 model."""
    
    # Get verifier config
    config = AutoConfig.from_pretrained(verifier_name, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    verifier_vocab_size = config.vocab_size
    
    # Get draft vocab size
    if draft_vocab_size is None:
        if d2t_path:
            d2t = np.load(d2t_path)
            draft_vocab_size = len(d2t)
        else:
            draft_vocab_size = 32000  # Default assumption
    
    head_dim = hidden_size // num_attention_heads
    
    print(f"=== Model Configuration ===")
    print(f"Verifier: {verifier_name}")
    print(f"Hidden size: {hidden_size}")
    print(f"Intermediate size: {intermediate_size}")
    print(f"Num attention heads: {num_attention_heads}")
    print(f"Num KV heads: {num_key_value_heads}")
    print(f"Head dim: {head_dim}")
    print(f"Verifier vocab size: {verifier_vocab_size}")
    print(f"Draft vocab size: {draft_vocab_size}")
    print(f"Num layers: {num_layers}")
    print()
    
    # === Trainable Parameters ===
    
    # FC layer: 3 * hidden_size -> hidden_size
    fc_params = 3 * hidden_size * hidden_size
    
    # First layer (Eagle3 first layer with hidden_norm)
    # Input is 2 * hidden_size (input_embeds + hidden_states concatenated)
    first_layer_input = 2 * hidden_size
    
    # Q projection: 2*hidden -> num_heads * head_dim
    q_proj_first = first_layer_input * (num_attention_heads * head_dim)
    # K projection: 2*hidden -> num_kv_heads * head_dim
    k_proj_first = first_layer_input * (num_key_value_heads * head_dim)
    # V projection: 2*hidden -> num_kv_heads * head_dim
    v_proj_first = first_layer_input * (num_key_value_heads * head_dim)
    # O projection: num_heads * head_dim -> hidden
    o_proj_first = (num_attention_heads * head_dim) * hidden_size
    
    # MLP first layer (input is hidden_size after o_proj)
    gate_proj_first = hidden_size * intermediate_size
    up_proj_first = hidden_size * intermediate_size
    down_proj_first = intermediate_size * hidden_size
    
    # LayerNorms in first layer (input_layernorm, post_attention_layernorm, hidden_norm)
    layernorm_first = 3 * hidden_size
    
    first_layer_params = (
        q_proj_first + k_proj_first + v_proj_first + o_proj_first +
        gate_proj_first + up_proj_first + down_proj_first +
        layernorm_first
    )
    
    # Regular decoder layers (layers 1 to num_layers-1)
    # Input is hidden_size
    q_proj = hidden_size * (num_attention_heads * head_dim)
    k_proj = hidden_size * (num_key_value_heads * head_dim)
    v_proj = hidden_size * (num_key_value_heads * head_dim)
    o_proj = (num_attention_heads * head_dim) * hidden_size
    
    gate_proj = hidden_size * intermediate_size
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    
    # LayerNorms (input_layernorm, post_attention_layernorm)
    layernorm = 2 * hidden_size
    
    regular_layer_params = (
        q_proj + k_proj + v_proj + o_proj +
        gate_proj + up_proj + down_proj +
        layernorm
    )
    
    # Final norm
    final_norm_params = hidden_size
    
    # LM head (trainable): hidden -> draft_vocab
    lm_head_params = hidden_size * draft_vocab_size
    
    # Total trainable
    trainable_params = (
        fc_params +
        first_layer_params +
        (num_layers - 1) * regular_layer_params +
        final_norm_params +
        lm_head_params
    )
    
    # === Frozen Parameters ===
    
    # Embeddings: verifier_vocab -> hidden
    embed_params = verifier_vocab_size * hidden_size
    
    # Verifier LM head (frozen): hidden -> draft_vocab
    verifier_lm_head_params = hidden_size * draft_vocab_size
    
    frozen_params = embed_params + verifier_lm_head_params
    
    total_params = trainable_params + frozen_params
    
    print(f"=== Parameter Breakdown ===")
    print(f"FC layer:              {fc_params:>15,}")
    print(f"First decoder layer:   {first_layer_params:>15,}")
    print(f"Regular layers (x{num_layers-1}):  {(num_layers-1) * regular_layer_params:>15,}")
    print(f"Final norm:            {final_norm_params:>15,}")
    print(f"LM head (trainable):   {lm_head_params:>15,}")
    print(f"{'─' * 40}")
    print(f"TRAINABLE TOTAL:       {trainable_params:>15,}")
    print()
    print(f"Embeddings (frozen):   {embed_params:>15,}")
    print(f"Verifier LM head:      {verifier_lm_head_params:>15,}")
    print(f"{'─' * 40}")
    print(f"FROZEN TOTAL:          {frozen_params:>15,}")
    print()
    print(f"{'═' * 40}")
    print(f"GRAND TOTAL:           {total_params:>15,}")
    print(f"Trainable:             {trainable_params / 1e6:>12.2f} M")
    print(f"Total:                 {total_params / 1e6:>12.2f} M")
    
    return trainable_params, total_params


def main():
    parser = argparse.ArgumentParser(description="Count Eagle3 model parameters")
    parser.add_argument(
        "--verifier-name-or-path",
        default="arcee-ai/trinity-mini",
        help="Verifier model name or path",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "--d2t-path",
        default=None,
        help="Path to d2t.npy for draft vocab size",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Draft vocabulary size (overrides d2t-path)",
    )
    
    args = parser.parse_args()
    
    count_params(
        verifier_name=args.verifier_name_or_path,
        num_layers=args.num_layers,
        draft_vocab_size=args.draft_vocab_size,
        d2t_path=args.d2t_path,
    )


if __name__ == "__main__":
    main()
