"""
Configuration printing utilities for mmExpert training scripts.
"""

from src.logger import log_message


def print_core_config(args, log_dir, cfg):
    """
    Print core configuration parameters with formatted output.

    Args:
        args: Command line arguments
        log_dir: Log directory path
        cfg: Configuration dictionary
    """
    log_message("TRAINING_CONFIG", "=" * 50, color="cyan")
    log_message("TRAINING_CONFIG", "[TRAINING CONFIGURATION]", color="cyan")
    log_message("TRAINING_CONFIG", "-" * 50, color="cyan")
    log_message("TRAINING_CONFIG", f"[EXPERIMENT NAME]      : {args.version}", color="green")
    log_message("TRAINING_CONFIG", f"[LOG DIRECTORY]        : {log_dir}", color="green")
    log_message("TRAINING_CONFIG", f"[DISTRIBUTED STRATEGY] : {args.strategy}", color="green")
    log_message("TRAINING_CONFIG", f"[WORLD SIZE]           : {args.world_size}", color="green")
    log_message("TRAINING_CONFIG", f"[RANDOM SEED]          : {args.seed}", color="green")
    log_message("TRAINING_CONFIG", f"[DATA CONFIG]          : {args.data_config}", color="green")
    log_message("TRAINING_CONFIG", f"[MODEL CONFIG]         : {args.model_config}", color="green")

    log_message("DATA_CONFIG", "\n[DATA CONFIGURATION]", color="cyan")
    log_message("DATA_CONFIG", "-" * 50, color="cyan")
    log_message("DATA_CONFIG", f"[BATCH SIZE]           : {cfg.data_cfg.params.cfg.batch_size}", color="green")
    log_message("DATA_CONFIG", f"[NUM WORKERS]          : {cfg.data_cfg.params.cfg.num_workers}", color="green")
    log_message("DATA_CONFIG", f"[MAX MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.max_motion_length}", color="green")
    log_message("DATA_CONFIG", f"[MIN MOTION LENGTH]    : {cfg.data_cfg.params.cfg.opt.min_motion_len}", color="green")
    log_message("DATA_CONFIG", f"[MAX TEXT LENGTH]       : {cfg.data_cfg.params.cfg.opt.max_text_len}", color="green")
    log_message("DATA_CONFIG", f"[UNIT LENGTH]          : {cfg.data_cfg.params.cfg.opt.unit_length}", color="green")
    log_message("DATA_CONFIG", f"[NORMALIZE]            : {cfg.data_cfg.params.cfg.opt.normalize}", color="green")
    log_message("DATA_CONFIG", f"[RADAR VIEWS]          : {cfg.data_cfg.params.cfg.opt.radar_views}", color="green")

    log_message("CLIP_CONFIG", "\n[CLIP CORE CONFIGURATION]", color="cyan")
    log_message("CLIP_CONFIG", "-" * 50, color="cyan")
    log_message("CLIP_CONFIG", f"[MAX EPOCHS]           : {cfg.model_cfg.params.max_epochs}", color="green")
    log_message("CLIP_CONFIG", f"[LEARNING RATE]       : {cfg.model_cfg.params.learning_rate}", color="green")
    log_message("CLIP_CONFIG", f"[TEMPERATURE]         : {cfg.model_cfg.params.temperature}", color="green")

    # Check if SigLIP mode is enabled
    if cfg.model_cfg.params.get('use_siglip', False):
        log_message("CLIP_CONFIG", "[MODEL TYPE]           : SigLIP", color="yellow")
        log_message("CLIP_CONFIG", f"[USE SIGLIP]           : {cfg.model_cfg.params.use_siglip}", color="yellow")
        log_message("CLIP_CONFIG", f"[LOSS DIST IMPLEMENTATION]: {cfg.model_cfg.params.get('loss_dist_impl', 'N/A')}", color="yellow")
    else:
        log_message("CLIP_CONFIG", "[MODEL TYPE]           : CLIP", color="green")

    log_message("TEXT_ENCODER_CONFIG", "\n[TEXT ENCODER CONFIGURATION]", color="cyan")
    log_message("TEXT_ENCODER_CONFIG", "-" * 50, color="cyan")

    # Extract text encoder configuration (use encoder_configs for both CLIP and SigLIP)
    encoder_configs = cfg.model_cfg.params.encoder_configs
    if 'text' in encoder_configs:
        text_cfg = encoder_configs.text
        log_message("TEXT_ENCODER_CONFIG", f"[MODEL NAME]           : {text_cfg.model_name}", color="green")
        log_message("TEXT_ENCODER_CONFIG", f"[EMBED DIM]            : {text_cfg.embed_dim}", color="green")
        log_message("TEXT_ENCODER_CONFIG", f"[MAX LENGTH]           : {text_cfg.max_length}", color="green")
        log_message("TEXT_ENCODER_CONFIG", f"[POOLING STRATEGY]     : {text_cfg.pooling_strategy}", color="green")
        log_message("TEXT_ENCODER_CONFIG", f"[FREEZE BACKBONE]      : {text_cfg.freeze_backbone}", color="green")
        if 'unfreeze_last_layers' in text_cfg:
            log_message("TEXT_ENCODER_CONFIG", f"[UNFREEZE LAYERS]      : {text_cfg.unfreeze_last_layers}", color="green")

    log_message("RADAR_ENCODER_CONFIG", "\n[RADAR ENCODER CONFIGURATION]", color="cyan")
    log_message("RADAR_ENCODER_CONFIG", "-" * 50, color="cyan")

    # Standard radar configuration for both CLIP and SigLIP
    if 'radar' in encoder_configs:
        radar_cfg = encoder_configs.radar
        log_message("RADAR_ENCODER_CONFIG", f"[ENCODER TYPE]         : {radar_cfg.get('encoder_type', 'N/A')}", color="green")
        log_message("RADAR_ENCODER_CONFIG", f"[EMBED DIM]            : {radar_cfg.embed_dim}", color="green")
        log_message("RADAR_ENCODER_CONFIG", f"[DROPOUT]             : {radar_cfg.dropout}", color="green")
        if 'vit_model' in radar_cfg:
            log_message("RADAR_ENCODER_CONFIG", f"[VIT MODEL]            : {radar_cfg.vit_model}", color="green")
            log_message("RADAR_ENCODER_CONFIG", f"[PRETRAINED]           : {radar_cfg.pretrained}", color="green")
            log_message("RADAR_ENCODER_CONFIG", f"[FREEZE BACKBONE]      : {radar_cfg.freeze_backbone}", color="green")
            log_message("RADAR_ENCODER_CONFIG", f"[PATCH SIZE RANGE]     : {radar_cfg.patch_size_range}", color="green")
            log_message("RADAR_ENCODER_CONFIG", f"[MAX SEQUENCE LENGTH]  : {radar_cfg.max_sequence_length}", color="green")

    log_message("CONFIG_END", "=" * 50, color="cyan")
    log_message("CONFIG_END", "", color="cyan")