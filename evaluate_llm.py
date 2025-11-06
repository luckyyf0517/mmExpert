"""Evaluation script for the new conversation template architecture"""

import argparse
import torch
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
import json
import os
from tqdm import tqdm
from termcolor import colored

from src.llm.llm.model_factory import ModelFactory
from src.llm.utils.config_loader import load_yaml_config
from src.llm.utils.conversation_utils import ConversationHandler
from src.llm.datamodule import WaveLLMDataModule


class WaveLLMEvaluator:
    """Evaluator for mmwave+LLM models using conversation templates"""

    def __init__(self, model_path, encoder_path, config=None):
        """
        Initialize evaluator

        Args:
            model_path: Path to fine-tuned model checkpoint
            encoder_path: Path to radar encoder
            config: Model configuration
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.config = config or self._load_default_config()

        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # Initialize conversation handler
        self.conv_handler = ConversationHandler(
            conv_template=self.config.get('conversation_template', 'conv_phi3'),
            wave_indicator=self.config.get('wave_indicator', '<wave>')
        )

    def _load_default_config(self):
        """Load default configuration"""
        return {
            'conversation_template': 'conv_phi3',
            'mm_use_wave_start_end': True,
            'wave_token_len': 248,
            'wave_indicator': '<wave>',
            'model_path': self.model_path
        }

    def _load_model(self):
        """Load the fine-tuned model"""
        # Load Lightning checkpoint
        model = LightningModule.load_from_checkpoint(self.model_path)
        model.eval()
        return model.cuda()

    def _load_tokenizer(self):
        """Load tokenizer from model configuration"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_path'],
            model_max_length=self.config.get('model_max_length', 2048),
            padding_side="right",
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def format_question_with_wave(self, question):
        """Format question with wave indicator"""
        return self.conv_handler.format_qa_without_wave(question, "[ANSWER_PLACEHOLDER]")

    def prepare_inputs(self, questions, mmwave_features):
        """Prepare inputs for generation"""
        batch_inputs = []
        batch_mmwave = []

        for question, features in zip(questions, mmwave_features):
            # Format conversation
            conversation_text = self.conv_handler.format_qa_with_wave_patch(
                question, "[ANSWER_PLACEHOLDER]"
            )
            conversation_text = self.conv_handler.replace_wave_indicator_with_tokens(conversation_text)

            # Tokenize
            tokenized = self.tokenizer(
                conversation_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )

            batch_inputs.append({
                'input_ids': tokenized.input_ids.squeeze(0),
                'attention_mask': tokenized.attention_mask.squeeze(0),
                'mmwave_features': features
            })

        return batch_inputs

    def generate_response(self, question, mmwave_features, max_new_tokens=50,
                         temperature=0.7, do_sample=True, top_p=0.9):
        """Generate response for a single question"""

        # Prepare inputs
        batch_inputs = self.prepare_inputs([question], [mmwave_features])
        inputs = batch_inputs[0]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'].unsqueeze(0).cuda(),
                mmwave_embeds=inputs['mmwave_features'].unsqueeze(0).cuda(),
                attention_mask=inputs['attention_mask'].unsqueeze(0).cuda(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response
        input_length = inputs['input_ids'].shape[0]
        response_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response.strip()

    def evaluate_dataset(self, dataset, output_file, batch_size=4):
        """Evaluate on entire dataset"""
        results = {}

        print(f"Evaluating {len(dataset)} samples...")

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]

            questions = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]
            mmwave_features = [item['wave_embed'] for item in batch]

            # Generate responses for batch
            responses = []
            for j, (question, features) in enumerate(zip(questions, mmwave_features)):
                response = self.generate_response(question, features)
                responses.append(response)

                # Print sample
                if i + j < 5:  # Print first 5 samples
                    print(colored(f"\nSample {i+j}:", "cyan"))
                    print(colored("Question:", "blue"), question)
                    print(colored("Response:", "green"), response)
                    print(colored("Ground Truth:", "yellow"), answers[j])
                    print("-" * 50)

            # Save results
            for j, (question, answer, response) in enumerate(zip(questions, answers, responses)):
                results[f"sample_{i+j}"] = {
                    "question": question,
                    "answer": answer,
                    "prediction": response
                }

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_file}")
        return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate mmwave+LLM model")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file for results")

    # Optional arguments
    parser.add_argument("--split", type=str, default="test_QAs",
                        help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")

    args = parser.parse_args()

    # Check if model checkpoint exists
    if not os.path.exists(args.model_checkpoint):
        print(colored("[ERROR]", "red") + f" Model checkpoint not found: {args.model_checkpoint}")
        return

    # Initialize evaluator
    print(f"Loading model from {args.model_checkpoint}")
    evaluator = WaveLLMEvaluator(
        model_path=args.model_checkpoint,
        encoder_path=args.data_root  # Assuming encoder is in data_root
    )

    # Load dataset
    print(f"Loading dataset from {args.data_root}")
    from src.llm.dataset import WaveCaptionDataset
    dataset = WaveCaptionDataset(
        data_root=args.data_root,
        split=args.split,
        tokenizer=evaluator.tokenizer
    )

    # Limit dataset size if specified
    if args.num_samples is not None:
        dataset = dataset[:args.num_samples]
        print(f"Limited to {len(dataset)} samples")

    # Evaluate
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        output_file=args.output_file,
        batch_size=args.batch_size
    )

    print(f"Evaluation completed! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()