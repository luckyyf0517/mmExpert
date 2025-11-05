"""
Simplified WaveLLM training script without complex configuration system.
Uses direct parameters for easier customization and maintenance.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import transformers

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.trainer.simple_training import SimpleTrainingPipeline
from src.wavellm import conversation as conversation_lib

# Configure environment
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def setup_logging(log_file: str = "train.log"):
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_environment():
    """Setup training environment."""
    # Disable torch initialization for faster startup
    def disable_torch_init():
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    disable_torch_init()

    # Setup conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates["conv_phi3"]


def validate_inputs(model_path: str, data_root: str, output_dir: str):
    """Validate input parameters."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


def main():
    """Main training function."""
    try:
        # Setup environment
        setup_environment()

        # Setup logging
        os.makedirs("logs", exist_ok=True)
        logger = setup_logging("logs/train_wavellm.log")

        # Parse command line arguments
        parser = transformers.HfArgumentParser((
            str,  # model_path
            str,  # data_root
            str,  # output_dir
            bool,  # flash_attention
            bool,  # use_lora
            int,   # lora_r
            int,   # lora_alpha
            float, # learning_rate
            int,   # batch_size
            float, # num_epochs
            bool,  # debug_mode
            bool,  # use_bf16
        ))

        parser.add_argument("--model_path", type=str, required=True,
                          help="Path to the model")
        parser.add_argument("--data_root", type=str, required=True,
                          help="Root directory for data")
        parser.add_argument("--output_dir", type=str, required=True,
                          help="Output directory for training")
        parser.add_argument("--flash_attention", action="store_true", default=True,
                          help="Use flash attention")
        parser.add_argument("--use_lora", action="store_true", default=True,
                          help="Use LoRA fine-tuning")
        parser.add_argument("--lora_r", type=int, default=64,
                          help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=16,
                          help="LoRA alpha")
        parser.add_argument("--learning_rate", type=float, default=2e-5,
                          help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=4,
                          help="Batch size")
        parser.add_argument("--num_epochs", type=float, default=3.0,
                          help="Number of training epochs")
        parser.add_argument("--debug_mode", action="store_true",
                          help="Enable debug mode")
        parser.add_argument("--use_bf16", action="store_true", default=True,
                          help="Use bfloat16 training")

        args = parser.parse_args()

        # Log training parameters
        logger.info("=" * 50)
        logger.info("WAVE LLM TRAINING - SIMPLIFIED ARCHITECTURE")
        logger.info("=" * 50)
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Epochs: {args.num_epochs}")
        logger.info(f"LoRA enabled: {args.use_lora}")
        if args.use_lora:
            logger.info(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
        logger.info(f"Flash attention: {args.flash_attention}")
        logger.info(f"Debug mode: {args.debug_mode}")
        logger.info(f"BF16: {args.use_bf16}")
        logger.info("=" * 50)

        # Validate inputs
        validate_inputs(args.model_path, args.data_root, args.output_dir)

        # Print system info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")

        # Create training pipeline
        logger.info("Creating training pipeline...")
        pipeline = SimpleTrainingPipeline(
            model_path=args.model_path,
            data_root=args.data_root,
            output_dir=args.output_dir,
            flash_attention=args.flash_attention,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            debug_mode=args.debug_mode,
            use_bf16=args.use_bf16
        )

        # Run training
        logger.info("Starting training...")
        results = pipeline.run_training()

        # Log results
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        if hasattr(results, 'training_loss'):
            logger.info(f"Final loss: {results.training_loss:.6f}")
        if hasattr(results, 'global_step'):
            logger.info(f"Total steps: {results.global_step}")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("=" * 50)

        return results

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()