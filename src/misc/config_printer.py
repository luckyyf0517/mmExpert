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
    print("=" * 80)
    print("[TRAINING CONFIGURATION]")
    print("-" * 80)
    print(f"[EXPERIMENT NAME]      : {args.version}")
    print(f"[LOG DIRECTORY]        : {log_dir}")
    print(f"[DISTRIBUTED STRATEGY] : {args.strategy}")
    print(f"[WORLD SIZE]           : {args.world_size}")
    print(f"[RANDOM SEED]          : {args.seed}")
    print(f"[DATA CONFIG]          : {args.data_config}")
    print(f"[MODEL CONFIG]         : {args.model_config}")

    print("\n[DATA CONFIGURATION]")
    print("-" * 80)
    print(f"[BATCH SIZE]           : {cfg.data_cfg.params.cfg.batch_size}")
    print(f"[NUM WORKERS]          : {cfg.data_cfg.params.cfg.num_workers}")
    print(f"[MAX MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.max_motion_length}")
    print(f"[MIN MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.min_motion_len}")
    print(f"[MAX TEXT LENGTH]       : {cfg.data_cfg.params.cfg.opt.max_text_len}")
    print(f"[UNIT LENGTH]          : {cfg.data_cfg.params.cfg.opt.unit_length}")
    print(f"[NORMALIZE]            : {cfg.data_cfg.params.cfg.opt.normalize}")
    print(f"[RADAR VIEWS]          : {cfg.data_cfg.params.cfg.opt.radar_views}")

    print("\n[CLIP CORE CONFIGURATION]")
    print("-" * 80)
    print(f"[MAX EPOCHS]           : {cfg.model_cfg.params.max_epochs}")
    print(f"[LEARNING RATE]       : {cfg.model_cfg.params.learning_rate}")
    print(f"[TEMPERATURE]         : {cfg.model_cfg.params.temperature}")

    # Check if SigLIP mode is enabled
    if cfg.model_cfg.params.get('use_siglip', False):
        print(f"[MODEL TYPE]           : SigLIP")
        print(f"[USE SIGLIP]           : {cfg.model_cfg.params.use_siglip}")
        print(f"[ADAPTIVE PATCH SIZE]  : {cfg.model_cfg.params.encoder_cfg.adaptive_patch_size}")
        print(f"[RANGE RESOLUTION]     : {cfg.model_cfg.params.encoder_cfg.range_resolution}")
        print(f"[DOPPLER RESOLUTION]   : {cfg.model_cfg.params.encoder_cfg.doppler_resolution}")
        print(f"[AZIMUTH RESOLUTION]   : {cfg.model_cfg.params.encoder_cfg.azimuth_resolution}")
        print(f"[FUSION METHOD]        : {cfg.model_cfg.params.encoder_cfg.fusion_method}")
        print(f"[CONTEXT LENGTH]       : {cfg.model_cfg.params.context_length}")
        print(f"[TRANSFORMER WIDTH]    : {cfg.model_cfg.params.transformer_width}")
        print(f"[TRANSFORMER HEADS]    : {cfg.model_cfg.params.transformer_heads}")
        print(f"[TRANSFORMER LAYERS]   : {cfg.model_cfg.params.transformer_layers}")
    else:
        print(f"[MODEL TYPE]           : CLIP")

    print("\n[TEXT ENCODER CONFIGURATION]")
    print("-" * 80)

    # Extract text encoder configuration based on model type
    if cfg.model_cfg.params.get('use_siglip', False):
        text_cfg = cfg.model_cfg.params.text_cfg
        print(f"[MODEL NAME]           : {text_cfg.model_name}")
        print(f"[EMBED DIM]            : {text_cfg.embed_dim}")
        print(f"[TEXT POOLING]         : {text_cfg.text_pooling}")
        print(f"[UNFREEZE LAYERS]      : {text_cfg.unfreeze_last_layer_num}")
    else:
        # Standard CLIP configuration
        encoder_configs = cfg.model_cfg.params.encoder_configs
        if 'text' in encoder_configs:
            text_cfg = encoder_configs.text
            print(f"[MODEL NAME]           : {text_cfg.model_name}")
            print(f"[EMBED DIM]            : {text_cfg.embed_dim}")
            print(f"[MAX LENGTH]           : {text_cfg.max_length}")
            print(f"[POOLING STRATEGY]     : {text_cfg.pooling_strategy}")
            print(f"[FREEZE BACKBONE]      : {text_cfg.freeze_backbone}")

    print("\n[RADAR ENCODER CONFIGURATION]")
    print("-" * 80)

    if cfg.model_cfg.params.get('use_siglip', False):
        radar_cfg = cfg.model_cfg.params.encoder_cfg
        print(f"[MODEL NAME]           : {radar_cfg.model_name}")
        print(f"[EMBED DIM]            : {radar_cfg.embed_dim}")
        print(f"[RADAR VIEWS]          : {radar_cfg.radar_views}")
        print(f"[PRETRAINED]           : {radar_cfg.pretrained}")
    else:
        # Standard CLIP radar configuration
        if 'radar' in encoder_configs:
            radar_cfg = encoder_configs.radar
            print(f"[EMBED DIM]            : {radar_cfg.embed_dim}")
            print(f"[DROPOUT]             : {radar_cfg.dropout}")

    print("=" * 80)
    print()