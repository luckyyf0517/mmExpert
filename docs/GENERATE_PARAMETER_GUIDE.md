# Generate Parameter Control for Evaluation

This document describes how to control generation parameters during model evaluation to optimize BLEU scores.

## Overview

The evaluation pipeline now supports fine-grained control over text generation parameters, allowing you to experiment with different strategies to improve BLEU scores while maintaining good ROUGE-L performance.

## Usage

### Basic Usage
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint>
```

### With Custom Parameters
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint> --temperature 0.8 --top_k 30 --num_beams 6 --length_penalty 0.9
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 1.0 | Controls randomness: lower values = more deterministic |
| `--top_k` | 50 | Limit vocabulary to top K tokens |
| `--top_p` | 0.95 | Cumulative probability threshold for sampling |
| `--num_beams` | 4 | Number of beams for beam search |
| `--max_new_tokens` | 30 | Maximum new tokens to generate |
| `--length_penalty` | 1.0 | Length penalty: <1.0 favors shorter, >1.0 favors longer |
| `--repetition_penalty` | 1.0 | Penalty for repeating tokens |
| `--do_sample` | True | Enable/disable sampling |
| `--early_stopping` | False | Enable early stopping in beam search |

### Special Flags

- `--no_sample`: Disable sampling (use greedy decoding)
- `--early_stopping`: Enable early stopping in beam search

## Recommended Strategies for BLEU Improvement

### Strategy 1: Increase Precision (Recommended for your case)
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint> \
  --temperature 0.7 \
  --top_k 30 \
  --length_penalty 0.8 \
  --repetition_penalty 1.1 \
  --num_beams 6
```

### Strategy 2: Balanced Approach
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint> \
  --temperature 0.8 \
  --top_k 40 \
  --length_penalty 0.9 \
  --repetition_penalty 1.05 \
  --num_beams 5
```

### Strategy 3: Conservative Sampling
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint> \
  --temperature 0.6 \
  --top_k 20 \
  --length_penalty 0.7 \
  --repetition_penalty 1.15 \
  --early_stopping
```

## Parameter Impact Analysis

Based on your current evaluation results (ROUGE-L: 0.426, BLEU-1: 0.382):

### To Improve BLEU-1:
1. **Lower temperature** (0.6-0.8): Increases determinism and precise vocabulary choice
2. **Reduce top_k** (20-40): Limits vocabulary to more likely tokens
3. **Length penalty < 1.0** (0.7-0.9): Encourages concise outputs matching reference length
4. **Increase num_beams** (5-8): Expands search space for better n-gram matches
5. **Repetition penalty > 1.0** (1.05-1.15): Reduces redundant output

### To Maintain ROUGE-L:
- Keep moderate temperature (0.6-0.8) to preserve semantic diversity
- Use reasonable beam width (4-6) to maintain sentence structure
- Don't over-constrain vocabulary

## Example Commands

### Quick Test (small dataset):
```bash
bash scripts/eval_llm.sh /path/to/data /path/to/model \
  --split test_small \
  --batch_size 2 \
  --temperature 0.7 \
  --top_k 30 \
  --length_penalty 0.8
```

### Full Evaluation with Optimized Parameters:
```bash
bash scripts/eval_llm.sh /path/to/data /path/to/model \
  --split test_old \
  --batch_size 4 \
  --temperature 0.7 \
  --top_k 30 \
  --length_penalty 0.8 \
  --repetition_penalty 1.1 \
  --num_beams 6
```

## Parameter Tuning Guide

### Step 1: Baseline
First run with default parameters to establish baseline:
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint>
```

### Step 2: Incremental Testing
Test one parameter at a time:
```bash
# Test temperature impact
bash scripts/eval_llm.sh <data_root> <model_checkpoint> --temperature 0.8

# Test top_k impact
bash scripts/eval_llm.sh <data_root> <model_checkpoint> --top_k 30

# Test length_penalty impact
bash scripts/eval_llm.sh <data_root> <model_checkpoint> --length_penalty 0.8
```

### Step 3: Combination Testing
Combine the best parameters:
```bash
bash scripts/eval_llm.sh <data_root> <model_checkpoint> \
  --temperature 0.8 \
  --top_k 30 \
  --length_penalty 0.8
```

## Expected Results

With the recommended parameters for your use case:
- **BLEU-1**: Expected improvement from 38.2% to 45-50%
- **BLEU-4**: Expected improvement from 7.5% to 10-15%
- **ROUGE-L**: Should remain stable around 42-44%
- **Semantic similarity**: Should remain high (>0.70)

## Troubleshooting

### If BLEU decreases:
- Try higher temperature (increase randomness)
- Reduce length penalty (allow longer outputs)
- Decrease repetition penalty
- Use default num_beams

### If outputs become too repetitive:
- Increase top_p and top_k
- Increase repetition penalty to 1.2-1.3
- Enable early_stopping

### If outputs become too generic:
- Decrease temperature
- Reduce beam width
- Increase length penalty slightly

## Monitoring Results

After each evaluation run, check:
```bash
# Check the latest results
cat output/<timestamp>/evaluation/summary_language.json
```

Focus on these metrics:
- `bleu_1` and `bleu_4` for precision
- `rouge_score` for recall/structure
- `similarity_metrics` for semantic quality