"""
Configuration printing utilities for mmExpert training scripts.
"""


def print_core_config(args, log_dir, cfg):
    """
    Print core configuration parameters with formatted output.

    Args:
        args: Command line arguments
        log_dir: Log directory path
        cfg: Configuration dictionary
    """
    print("=" * 50)
    print("[TRAINING CONFIGURATION]")
    print("-" * 50)
    print(f"[EXPERIMENT NAME]      : {args.version}")
    print(f"[LOG DIRECTORY]        : {log_dir}")
    print(f"[DISTRIBUTED STRATEGY] : {args.strategy}")
    print(f"[WORLD SIZE]           : {args.world_size}")
    print(f"[RANDOM SEED]          : {args.seed}")
    print(f"[DATA CONFIG]          : {args.data_config}")
    print(f"[MODEL CONFIG]         : {args.model_config}")

    print("\n[DATA CONFIGURATION]")
    print("-" * 50)
    print(f"[BATCH SIZE]           : {cfg.data_cfg.params.cfg.batch_size}")
    print(f"[NUM WORKERS]          : {cfg.data_cfg.params.cfg.num_workers}")
    print(f"[MAX MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.max_motion_length}")
    print(f"[MIN MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.min_motion_len}")
    print(f"[MAX TEXT LENGTH]       : {cfg.data_cfg.params.cfg.opt.max_text_len}")
    print(f"[UNIT LENGTH]          : {cfg.data_cfg.params.cfg.opt.unit_length}")
    print(f"[NORMALIZE]            : {cfg.data_cfg.params.cfg.opt.normalize}")
    print(f"[RADAR VIEWS]          : {cfg.data_cfg.params.cfg.opt.radar_views}")

    print("\n[CLIP CORE CONFIGURATION]")
    print("-" * 50)
    print(f"[MAX EPOCHS]           : {cfg.model_cfg.params.max_epochs}")
    print(f"[LEARNING RATE]       : {cfg.model_cfg.params.learning_rate}")
    print(f"[TEMPERATURE]         : {cfg.model_cfg.params.temperature}")

    # Check if SigLIP mode is enabled
    if cfg.model_cfg.params.get('use_siglip', False):
        print(f"[MODEL TYPE]           : SigLIP")
        print(f"[USE SIGLIP]           : {cfg.model_cfg.params.use_siglip}")
        print(f"[LOSS DIST IMPLEMENTATION]: {cfg.model_cfg.params.get('loss_dist_impl', 'N/A')}")
    else:
        print(f"[MODEL TYPE]           : CLIP")

    print("\n[TEXT ENCODER CONFIGURATION]")
    print("-" * 50)

    # Extract text encoder configuration (use encoder_configs for both CLIP and SigLIP)
    encoder_configs = cfg.model_cfg.params.encoder_configs
    if 'text' in encoder_configs:
        text_cfg = encoder_configs.text
        print(f"[MODEL NAME]           : {text_cfg.model_name}")
        print(f"[EMBED DIM]            : {text_cfg.embed_dim}")
        print(f"[MAX LENGTH]           : {text_cfg.max_length}")
        print(f"[POOLING STRATEGY]     : {text_cfg.pooling_strategy}")
        print(f"[FREEZE BACKBONE]      : {text_cfg.freeze_backbone}")
        if 'unfreeze_last_layers' in text_cfg:
            print(f"[UNFREEZE LAYERS]      : {text_cfg.unfreeze_last_layers}")

    print("\n[RADAR ENCODER CONFIGURATION]")
    print("-" * 50)

    # Standard radar configuration for both CLIP and SigLIP
    if 'radar' in encoder_configs:
        radar_cfg = encoder_configs.radar
        print(f"[ENCODER TYPE]         : {radar_cfg.get('encoder_type', 'N/A')}")
        print(f"[EMBED DIM]            : {radar_cfg.embed_dim}")
        print(f"[DROPOUT]             : {radar_cfg.dropout}")
        if 'vit_model' in radar_cfg:
            print(f"[VIT MODEL]            : {radar_cfg.vit_model}")
            print(f"[PRETRAINED]           : {radar_cfg.pretrained}")
            print(f"[FREEZE BACKBONE]      : {radar_cfg.freeze_backbone}")
            print(f"[PATCH SIZE RANGE]     : {radar_cfg.patch_size_range}")
            print(f"[MAX SEQUENCE LENGTH]  : {radar_cfg.max_sequence_length}")

    print("=" * 50)
    print()