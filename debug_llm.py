"""Debug script to test checkpoint on training dataset"""

import argparse
import os
os.chdir('/root/autodl-tmp/mmExpert')

import sys
sys.path.append('.')

import torch
from tqdm import tqdm

from src.llm.datamodule import WaveLLMDataModule
from src.llm.utils.common_utils import load_model_from_checkpoint
from src.logger import log_message

def main():
    parser = argparse.ArgumentParser(description="Debug checkpoint on training dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum number of samples to test")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    
    args = parser.parse_args()
    
    # Load model
    model, cfg = load_model_from_checkpoint(
        args.model_checkpoint,
        args.config,
        args.data_root
    )
    
    # Load training dataset - use training stage to get train_dataloader
    log_message("LOAD", f"Loading TRAINING dataset from {args.data_root}", color="cyan")
    cfg.data_cfg.batch_size = args.batch_size
    
    data_module = WaveLLMDataModule(cfg.data_cfg)
    data_module.setup(stage='fit')  # Use 'fit' stage to load training data
    train_dataloader = data_module.train_dataloader()
    
    # Test on training data
    log_message("PROGRESS", f"Testing on {args.max_samples} training samples", color="yellow")
    print("")
    
    total_samples = 0
    all_outputs = []
    all_questions = []
    all_answers = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Processing batches")):
            if total_samples >= args.max_samples:
                break
            
            # Move batch to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            mmwave_features = batch['mmwave_features']  # Already a dict
            labels = batch['labels'].cuda()
            questions = batch.get('questions', [''] * input_ids.shape[0])
            answers = batch.get('answers', [''] * input_ids.shape[0])
            
            batch_size = input_ids.shape[0]
            
            # Process each sample in batch
            for i in range(batch_size):
                if total_samples >= args.max_samples:
                    break
                
                # Extract single sample
                input_ids_single = input_ids[i:i+1]
                attention_mask_single = attention_mask[i:i+1]
                # Extract single sample from mmwave_features dict
                mmwave_features_single = {k: v[i:i+1].cuda() if torch.is_tensor(v) else v for k, v in mmwave_features.items()}
                labels_single = labels[i:i+1]
                
                # Process mmwave features
                mmwave_embeds_single = model._process_mmwave_features(mmwave_features_single)
                
                # Forward pass (no generation, just get loss and logits)
                outputs = model.model(
                    input_ids=input_ids_single,
                    input_wave_embeds=mmwave_embeds_single,
                    attention_mask=attention_mask_single,
                    labels=labels_single,
                    return_dict=True
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Get predicted tokens (greedy decoding from logits)
                pred_token_ids = torch.argmax(logits, dim=-1)

                # Find where labels are not -100 (actual answer part)
                answer_mask = labels_single[0] != -100
                if not answer_mask.any():
                    # No valid labels, skip this sample
                    continue

                # Also decode only the answer portion for comparison
                answer_positions = answer_mask.nonzero(as_tuple=True)[0]
                answer_start = answer_positions[0].item()
                answer_end = answer_positions[-1].item() + 1

                # For causal LM, the logits at position i predict the token at position i+1
                # So we look at logits[answer_start-1:answer_end-1] to match labels[answer_start:answer_end]
                if answer_start > 0:
                    pred_answer_ids = pred_token_ids[0, answer_start-1:answer_end-1]
                else:
                    pred_answer_ids = pred_token_ids[0, answer_start:answer_end]

                pred_text = model.tokenizer.decode(
                    pred_answer_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                # Decode ground truth answer
                gt_text = model.tokenizer.decode(
                    labels_single[0, answer_mask],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                all_outputs.append(pred_text)
                all_questions.append(questions[i] if i < len(questions) else "N/A")
                all_answers.append(gt_text)

                # Print loss for this sample
                log_message("PROGRESS", f"Sample {total_samples+1} - Loss: {loss.item():.4f}", color="yellow")
                
                total_samples += 1
    
    # Print results
    print("")
    log_message("RESULTS", "="*80, color="blue")
    log_message("RESULTS", "TRAINING SET FITTING TEST RESULTS", color="blue")
    log_message("RESULTS", "="*80, color="blue")
    print("")
    
    for idx, (question, output, answer) in enumerate(zip(all_questions, all_outputs, all_answers)):
        log_message("SAMPLE", f"Sample {idx+1}/{total_samples}", color="cyan")
        if question and question != "N/A":
            print(f"Question: {question}")
        print(f"Model Prediction (from logits): {output}")
        print(f"Ground Truth: {answer}")
        print("-"*80)
        print("")
    
    # Calculate simple metrics
    log_message("SUMMARY", "="*80, color="blue")
    log_message("SUMMARY", "SUMMARY", color="blue")
    log_message("SUMMARY", "="*80, color="blue")
    print(f"Total samples tested: {total_samples}")
    print(f"Average output length: {sum(len(o.split()) for o in all_outputs) / len(all_outputs):.1f} words")
    
    # Check if outputs are reasonable (not gibberish)
    reasonable_count = sum(1 for o in all_outputs if len(o.split()) >= 3 and not any(c in o for c in ['�', '##']))
    print(f"Reasonable outputs: {reasonable_count}/{total_samples} ({reasonable_count/total_samples*100:.1f}%)")
    
    print("")
    log_message("SUCCESS", "Debug test completed!", color="green")


if __name__ == "__main__":
    main()

