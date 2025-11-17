import argparse
import json
from collections import defaultdict
import re
import os
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm

from collections import OrderedDict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from scipy.spatial.distance import cosine
from glob import glob
import pickle
import copy

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from utils import (
    load_evaluation_data, 
    save_evaluation_results,
    clean_answer,
    special_token_filter,
    refined_EM
)
CLEAN_TEXT = True


class Evaluator():
    def __init__(self,directory_path,eval_bs) -> None:
        self.eval_bs = eval_bs
        self.directory_path = directory_path
        # Set cache directory for offline mode
        cache_dir = "/root/autodl-tmp/mmExpert/huggingface"

        self.simcse_tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large",
            cache_dir=cache_dir,
            local_files_only=False
        )
        self.simcse_model = AutoModel.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large",
            cache_dir=cache_dir,
            local_files_only=False
        ).to("cuda")

        self.sbert_model = SentenceTransformer('all-mpnet-base-v2',device="cuda")

    @staticmethod
    def to_coco(kvs, keys):
        res = defaultdict(list)
        for k in keys:
            if k in kvs:
                caps = kvs[k]
                for c in caps:
                    res[k].append({'caption': c})
            else:
                res[k].append({'caption': ''})
        return res

    def evaluate(self,ground_truths,prediction,verbose = True):

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),  # Commented out - requires Stanford CoreNLP
        ]
        tokenizer = PTBTokenizer()
        ref_sent = ground_truths
        hypo_sent = prediction
        final_scores = {}
        ref_coco = tokenizer.tokenize(self.to_coco(ref_sent, ref_sent.keys()))
        hypo_coco = tokenizer.tokenize(self.to_coco(hypo_sent, ref_sent.keys()))
        for scorer, method in scorers:
            if verbose:
                print('computing %s score...' % (scorer.method()))
            try:
                if hasattr(scorer, 'method') and scorer.method() == "Bleu":
                    score, scores = scorer.compute_score(ref_coco, hypo_coco, verbose=0)
                else:
                    score, scores = scorer.compute_score(ref_coco, hypo_coco)
                if type(score) == list:
                    for m, s in zip(method, score):
                        final_scores[m] = s
                else:
                    final_scores[method] = score
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to compute {method} score: {e}")
                    print("METEOR requires Java environment. Skipping METEOR metric.")
                # Skip this metric if it fails (e.g., METEOR without Java)
                continue
        return final_scores

    @staticmethod
    def print_formated_dict(lan):
        for key in lan:
            print(f"{key}:      {lan[key]}")

    def batch_eval(self,all_pred,all_gt,gt_count):
        """
        Args:
            gt_count(list): stores number of possible answers to a question
            all_pred(list): all prediction
            all_gt(list): all ground truth,   len(all_gt)>=len(all_pred)

        Return:
            tuple: all_sbert_sim,all_simcse_sim
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(all_pred+all_gt,show_progress_bar=False,device="cuda")
            inputs = self.simcse_tokenizer(all_pred+all_gt, padding=True, truncation=True, return_tensors="pt").to("cuda")
            simcse_embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        all_pred_sbert_embed = sbert_embeddings[:len_of_pred]
        all_pred_simcse_embed = simcse_embeddings[:len_of_pred]

        all_gt_sbert_embed = sbert_embeddings[len_of_pred:]
        all_gt_simcse_embed = simcse_embeddings[len_of_pred:]

        all_sbert_sim = []
        all_simcse_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            simcse_similarity = -100
            sbert_similarity = -100
            for j in range(accumulated,accumulated+gt_count[i]):
                sbert_similarity = max(sbert_similarity, util.cos_sim(all_pred_sbert_embed[i], 
                                                                        all_gt_sbert_embed[j])[0][0].item())
                simcse_similarity = max(simcse_similarity ,1 - cosine(all_pred_simcse_embed[i].cpu().detach().numpy(), 
                                                                        all_gt_simcse_embed[j].cpu().detach().numpy())) 
            all_sbert_sim.append(sbert_similarity)
            all_simcse_sim.append(simcse_similarity)
            accumulated+=gt_count[i]
        torch.cuda.empty_cache()
        return all_sbert_sim,all_simcse_sim

    def evaluate_with_best_gt(self, ground_truths, prediction, verbose=True):
        """
        Evaluate each prediction against all ground truths and take the best score for each metric.
        Efficient version that tokenizes all texts once upfront.
        
        Args:
            ground_truths: dict with keys as sample IDs and values as lists of ground truth texts
            prediction: dict with keys as sample IDs and values as lists of prediction texts
            verbose: whether to print progress
            
        Returns:
            dict: final scores where each metric is the average of best scores across all samples
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        tokenizer = PTBTokenizer()
        
        # Pre-tokenize ALL texts once upfront
        if verbose:
            print("Tokenizing all texts...")
        
        # Build comprehensive data structure for tokenization
        all_data = {}
        pred_data = {}
        
        for sample_id in ground_truths.keys():
            if sample_id not in prediction:
                continue
                
            # Add prediction
            pred_text = prediction[sample_id][0]
            pred_data[sample_id] = [{'caption': pred_text}]
            
            # Add all ground truths
            all_data[sample_id] = []
            for gt_text in ground_truths[sample_id]:
                all_data[sample_id].append({'caption': gt_text})
        
        # Tokenize everything once
        tokenized_all = tokenizer.tokenize(all_data)
        tokenized_pred = tokenizer.tokenize(pred_data)
        
        # Initialize score accumulators for each metric
        all_best_scores = defaultdict(list)
        
        # Process each sample
        for sample_id in tqdm(ground_truths.keys(), desc="Evaluating samples"):
            if sample_id not in prediction:
                continue
                
            gt_tokenized = tokenized_all[sample_id]  # All GT texts for this sample
            pred_tokenized = tokenized_pred[sample_id][0]  # Prediction text
            
            best_scores = {}
            
            # For each scorer, evaluate against all GT texts and take the best score
            for scorer, method in scorers:
                if verbose and sample_id == list(ground_truths.keys())[0]:  # Print only once
                    print('computing %s score...' % (scorer.method()))
                
                sample_best_score = -1
                sample_best_scores_list = {}  # For multi-score metrics
                
                # Create prediction coco format
                pred_coco = {sample_id: [pred_tokenized]}
                
                # Evaluate against each ground truth text
                for gt_tokenized_text in gt_tokenized:
                    gt_coco = {sample_id: [gt_tokenized_text]}
                    try:
                        if hasattr(scorer, 'method') and scorer.method() == "Bleu":
                            score, _ = scorer.compute_score(gt_coco, pred_coco, verbose=0)
                        else:
                            score, _ = scorer.compute_score(gt_coco, pred_coco)
                        
                        if isinstance(score, list):
                            # For metrics like BLEU that return multiple scores
                            for i, m in enumerate(method):
                                if m not in sample_best_scores_list:
                                    sample_best_scores_list[m] = -1
                                sample_best_scores_list[m] = max(sample_best_scores_list[m], score[i])
                        else:
                            # For single score metrics
                            sample_best_score = max(sample_best_score, score)
                    except Exception as e:
                        # Skip this GT text if scorer fails (e.g., METEOR without Java)
                        if verbose and sample_id == list(ground_truths.keys())[0]:
                            print(f"Warning: Failed to compute {method} for sample {sample_id}: {e}")
                        continue
                
                # Store the best scores
                if sample_best_scores_list:
                    # Multi-score metric
                    for metric, score in sample_best_scores_list.items():
                        best_scores[metric] = score
                elif sample_best_score >= 0:
                    # Single-score metric (only if we got a valid score)
                    best_scores[method] = sample_best_score
            
            # Add best scores to accumulators
            for metric, score in best_scores.items():
                all_best_scores[metric].append(score)
        
        # Calculate final average scores
        final_scores = {}
        for metric, scores in all_best_scores.items():
            final_scores[metric] = sum(scores) / len(scores) if scores else 0.0
            
        return final_scores


    def load_data_and_eval(self, max_length=1024, use_exist=False):
        all_pred = {}
        lan_gt = {}
        lan_pred = {}

        all_simcse_similarity = []
        all_sbert_similarity = []

        # Load evaluation data using the unified function
        all_pred = load_evaluation_data(self.directory_path, merged_filename="results.json", use_exist=use_exist)

        # Single-threaded evaluation
        bar = tqdm(all_pred)
        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []

        for idx, key in enumerate(bar):
            pred_text = all_pred[key]["prediction"]
            answer_text = all_pred[key]["answer"]
            # Answer should already be a list (preprocessed in utils.py)
            gt_list = answer_text if isinstance(answer_text, list) else [answer_text]

            pred = special_token_filter(pred_text, clean=CLEAN_TEXT, truncation=True, max_length=max_length)
            lan_pred[key] = [pred]
            lan_gt[key] = [special_token_filter(i, clean=CLEAN_TEXT, truncation=True, max_length=max_length) for i in gt_list]
            batch_lan_pred += lan_pred[key]
            batch_lan_gt += lan_gt[key]
            count_gt += [len(lan_gt[key])]
            if idx % self.eval_bs == 0:
                score = self.batch_eval(batch_lan_pred, batch_lan_gt, count_gt)
                all_simcse_similarity += score[1]
                all_sbert_similarity += score[0]

                batch_lan_pred = []
                batch_lan_gt = []
                count_gt = []

        if len(batch_lan_pred):
            score = self.batch_eval(batch_lan_pred, batch_lan_gt, count_gt)
            all_simcse_similarity += score[1]
            all_sbert_similarity += score[0]

        assert len(all_simcse_similarity) == len(all_pred)
        
        # Use the new evaluation method that takes best GT score for each sample
        final_scores = self.evaluate_with_best_gt(ground_truths=lan_gt,
                                                 prediction=lan_pred)

        # Calculate EM scores with best GT matching
        EM_best = []
        EM_refine_best = []
        
        for key in lan_pred:
            pred = lan_pred[key][0]
            gt_texts = lan_gt[key]
            
            # Calculate EM for each GT text and take the best
            best_EM = 0
            best_EM_refine = 0
            
            for gt_text in gt_texts:
                # Create temporary data structures for EM calculation
                temp_pred = {key: [pred]}
                temp_gt = {key: [gt_text]}
                
                EM_result = refined_EM(temp_pred, temp_gt, not_refine=True)
                EM_refine_result = refined_EM(temp_pred, temp_gt, not_refine=False)
                
                best_EM = max(best_EM, EM_result[0])
                best_EM_refine = max(best_EM_refine, EM_refine_result[0])
            
            EM_best.append(best_EM)
            EM_refine_best.append(best_EM_refine)
        
        # Print final results (without "(best)" summary_language
        print("=== Best GT Scores (Average across all samples) ===")
        print("Bleu_1:      ", final_scores.get('Bleu_1', 0.0))
        print("Bleu_2:      ", final_scores.get('Bleu_2', 0.0))
        print("Bleu_3:      ", final_scores.get('Bleu_3', 0.0))
        print("Bleu_4:      ", final_scores.get('Bleu_4', 0.0))
        if 'METEOR' in final_scores:
            print("METEOR:     ", final_scores['METEOR'])
        print("ROUGE_L:      ", final_scores.get('ROUGE_L', 0.0))
        print("CIDEr:      ", final_scores.get('CIDEr', 0.0))
        print("EM:         ", sum(EM_best)/len(EM_best))
        print("refined EM: ", sum(EM_refine_best)/len(EM_refine_best))
        print("simcse:     ", sum(all_simcse_similarity)/len(all_simcse_similarity))
        print("sbert:      ", sum(all_sbert_similarity)/len(all_sbert_similarity))

        # Return evaluation results for saving
        result = {
            'total_samples': len(EM_best),
            'bleu_scores': {
                'bleu_1': final_scores.get('Bleu_1', 0.0),
                'bleu_2': final_scores.get('Bleu_2', 0.0),
                'bleu_3': final_scores.get('Bleu_3', 0.0),
                'bleu_4': final_scores.get('Bleu_4', 0.0)
            },
            'rouge_score': final_scores.get('ROUGE_L', 0.0),
            'cider_score': final_scores.get('CIDEr', 0.0),
            'exact_match': {
                'em': sum(EM_best)/len(EM_best),
                'em_refined': sum(EM_refine_best)/len(EM_refine_best)
            },
            'similarity_metrics': {
                'simcse': sum(all_simcse_similarity)/len(all_simcse_similarity),
                'sbert': sum(all_sbert_similarity)/len(all_sbert_similarity)
            }
        }
        
        # Add METEOR score if available
        if 'METEOR' in final_scores:
            result['meteor_score'] = final_scores['METEOR']
        
        return result
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Language evaluation for motion descriptions')
    parser.add_argument('--evaluation_dir', type=str,
                       help='Path to evaluation directory containing results_rank_*.json files')
    parser.add_argument('--directory_path', type=str,
                       help='Path to directory or single JSON file (deprecated, use --evaluation_dir)')
    parser.add_argument('--eval_bs', type=int, default=100, help='evaluation batch size')
    parser.add_argument('--use_exist', action='store_true',
                       help='Use existing merged results.json file if it exists')
    parser.add_argument('--output_filename', type=str, default='summary_language.json',
                       help='Output filename for language evaluation results (default: summary_language.json)')

    args = parser.parse_args()

    # Handle both old and new parameter names
    if args.evaluation_dir:
        input_path = args.evaluation_dir
    elif args.directory_path:
        input_path = args.directory_path
    else:
        parser.error("Either --evaluation_dir or --directory_path must be provided")

    print(f"Running language evaluation for {input_path} ...")
    print("Using optimized text cleaning strategy for BLEU scores")

    eval = Evaluator(
        directory_path=input_path,
        eval_bs=args.eval_bs
    )
    results = eval.load_data_and_eval(max_length=1024, use_exist=args.use_exist)

    # Save evaluation results
    if os.path.isdir(input_path):
        output_file = os.path.join(input_path, args.output_filename)
    else:
        # For single file input, save in same directory
        output_file = os.path.join(os.path.dirname(input_path), args.output_filename)

    save_evaluation_results(results, output_file)
    print(f"\nLanguage evaluation completed successfully!")
    print(f"Results saved to: {output_file}")


        