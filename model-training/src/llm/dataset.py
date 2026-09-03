import json
import torch
import numpy as np
from copy import deepcopy
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import random
import yaml
import os.path as osp
from src.logger import log_message

import sys; sys.path.append('.')
from src.clip.dataset import load_radar_data, mix_static_overlay, resolve_radar_path

# Constants
IGNORE_INDEX = -100
DEFAULT_WAVE_INDICATOR = "<wave>"
DEFAULT_WAVE_PATCH_TOKEN = "<wave_patch>"
DEFAULT_WAVE_START_TOKEN = "<wave_bos>"
DEFAULT_WAVE_END_TOKEN = "<wave_eos>"
DEFAULT_WAVE_TOKEN_LEN = 248

# Default configuration for radar dataset
DEFAULT_REAL_CONFIG = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'normalize': 'per_frame',  # 'none', 'per_frame', 'global', or 'log'
    'radar_views': 'doppler',
}

# Default question prompts for radar data
DEFAULT_QUESTION_PROMPTS = [
    "Describe the human motion based on the provided micro-Doppler radar signal.",
    "Interpret the radar data and explain the corresponding human motion patterns.",
    "Analyze the given radar signals to describe the overall movement of the human body.",
    "Given radar data representing motion patterns, explain how a human is moving.",
    "Use the provided radar signals to describe the sequence of human body movements.",
    "Interpret the radar data to describe the motion pattern of the human body.",
    "Based on the radar signal data, explain the human motion being represented.",
    "Analyze the radar signals to generate a description of human motion.",
    "Use the radar data to interpret and describe the human body's movement.",
    "Given the radar signals, describe the motion of the human body in detail."
]

RADAR_LOAD_OVERRIDE_KEYS = {
    'udoppler_preprocess',
    'udoppler_preprocess_percentile',
    'udoppler_preprocess_gamma',
    'udoppler_post_threshold_gamma',
    'radar_npz_preprocessed',
    'normalize',
    'radar_views',
    'mmwave_postfix',
    'allow_radar_fallback',
    'udoppler_legacy_suffix',
    'real_preprocess',
    'real_view_transform',
}


class WaveCaptionDataset(Dataset):
    """Dataset for wave-language supervision with multimodal support."""

    def __init__(
        self,
        data_root=None,
        split="train",
        stage='train',
        tokenizer=None,
        caption_only=None,
        supervision_mode=None,
        caption_loading_mode=None,
        use_random_question_for_caption=True,
        static_bg_npz=None,
        range_drop_prob=0.0,
        doppler_drop_prob=0.0,
        azimuth_drop_prob=0.0,
    ) -> None:
        super(WaveCaptionDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.stage = stage
        self.tokenizer = tokenizer
        self.use_random_question_for_caption = use_random_question_for_caption
        self.opt = self._load_config(data_root, split)
        if static_bg_npz:
            self.opt.static_bg_npz = static_bg_npz
        for _pn in ('range_drop_prob', 'doppler_drop_prob', 'azimuth_drop_prob'):
            _pv = locals()[_pn]
            if _pv > 0:
                self.opt[_pn] = _pv
        self.sample_dataset = WaveSampleDataset(data_root=data_root, split=split, opt=self.opt)
        self.caption_task_weight = self.opt.get('caption_task_weight', None)
        self.qa_task_weight = self.opt.get('qa_task_weight', None)

        # Backward compatibility: caption_only is mapped to supervision_mode.
        if supervision_mode is None and caption_only is not None:
            supervision_mode = 'caption' if caption_only else 'qa'
        if supervision_mode is None:
            supervision_mode = self.opt.get('supervision_mode')
        if supervision_mode is None:
            supervision_mode = 'caption' if bool(self.opt.get('caption_only', False)) else 'qa'
        self.supervision_mode = supervision_mode
        self.caption_loading_mode = self._resolve_caption_loading_mode(caption_loading_mode)

        self.data = self._build_supervision_index()

        # On-the-fly static background overlay (configurable)
        self._static_views_cache: dict | None = None
        self._static_bg_npz: str | None = self.opt.get('static_bg_npz', None)
        if self._static_bg_npz:
            self._static_views_cache = self._load_static_views(self._static_bg_npz)

        log_message("LOAD", f"load {len(self.data)} data as {self.split} set", color="green")

    @staticmethod
    def _load_static_views(static_npz: str) -> dict[str, np.ndarray]:
        """Load and cache static background three-view NPZ."""
        if not osp.exists(static_npz):
            raise FileNotFoundError(f"static_bg_npz not found: {static_npz}")
        data = np.load(static_npz)
        views = {}
        for k in ("range_time", "doppler_time", "azimuth_time"):
            if k not in data:
                raise KeyError(f"static NPZ missing key '{k}'; available: {list(data.keys())}")
            views[k] = np.asarray(data[k], dtype=np.float32)
        log_message("LOAD", f"cached static bg views from {static_npz}", color="green")
        return views

    def _selected_radar_views(self, opt=None) -> list[str]:
        opt = opt or self.opt
        radar_views = opt.get('radar_views', 'all')
        view_selection = {
            'all': ['range_time', 'doppler_time', 'azimuth_time'],
            'doppler': ['doppler_time'],
            'udoppler': ['doppler_time'],
            'range': ['range_time'],
            'azimuth': ['azimuth_time'],
        }
        if radar_views not in view_selection:
            raise ValueError(f"Unknown radar_views configuration: {radar_views}")
        return view_selection[radar_views]

    def _load_config(self, data_root, split):
        """Load configuration based on split type."""
        if split == 'real':
            return edict(DEFAULT_REAL_CONFIG)
        else:
            config_path = osp.join(data_root, 'radar_encoder_config.yaml')
            try:
                with open(config_path, 'r') as file:
                    yaml_data = yaml.safe_load(file)
            except yaml.constructor.ConstructorError:
                from yaml import UnsafeLoader
                with open(config_path, 'r') as file:
                    yaml_data = yaml.load(file, Loader=UnsafeLoader)

            if yaml_data is None:
                raise ValueError(f"YAML config is empty: {config_path}")

            yaml_data = edict(yaml_data)

            # Try to get dataset_cfg, fallback to default if not found
            if 'dataset_cfg' in yaml_data and yaml_data.dataset_cfg is not None:
                return edict(yaml_data.dataset_cfg)
            else:
                # If dataset_cfg is not in YAML, return default config
                return edict(DEFAULT_REAL_CONFIG)

    def _resolve_caption_loading_mode(self, caption_loading_mode):
        """Resolve whether caption supervision should be expanded or sampled."""
        if caption_loading_mode is not None:
            mode = caption_loading_mode
        else:
            mode = self.opt.get('caption_loading_mode')

        if mode is None:
            mode = 'expand' if self.stage == 'train' else 'sample'

        if mode not in {'expand', 'sample'}:
            raise ValueError(f"Unsupported caption_loading_mode: {mode}")

        return mode

    def _build_caption_supervision(self, data_item):
        """Caption supervision is represented as QA with a synthetic question."""
        supervision_items = []
        for caption in data_item.get('captions', []):
            supervision_items.append({
                'task_type': 'caption',
                'qa_id': None,
                'question': None,
                'answer': caption,
            })
        return supervision_items

    def _build_question_supervision(self, data_item):
        """QA supervision from explicit question-answer annotations."""
        supervision_items = []
        if 'questions' not in data_item:
            return supervision_items

        for qa_id, qa_data in data_item['questions'].items():
            supervision_items.append({
                'task_type': 'qa',
                'qa_id': qa_id,
                'question': qa_data.get('question', ''),
                'answer': qa_data.get('answer', '')
            })
        return supervision_items

    def _select_supervision_items(self, data_item):
        """Expand a raw sample into supervision items according to supervision_mode."""
        caption_items = self._build_caption_supervision(data_item)
        qa_items = self._build_question_supervision(data_item)

        if self.supervision_mode == 'caption':
            return caption_items
        if self.supervision_mode == 'qa':
            return qa_items
        if self.supervision_mode == 'caption+qa':
            return caption_items + qa_items

        raise ValueError(f"Unsupported supervision_mode: {self.supervision_mode}")

    def _create_caption_sample_item(self, data_item):
        """Create one caption supervision item per raw sample with all references grouped."""
        return self._create_data_item(data_item, {
            'task_type': 'caption',
            'qa_id': None,
            'question': None,
            'answer': '#'.join(data_item.get('captions', [])),
        })

    def _build_caption_supervision_items_for_mode(self, data_item, caption_items):
        """Return caption supervision items respecting caption_loading_mode."""
        if self.caption_loading_mode == 'sample':
            return [{
                'task_type': 'caption',
                'qa_id': None,
                'question': None,
                'answer': '#'.join(data_item.get('captions', [])),
            }]
        return list(caption_items)

    def _append_caption_items(self, processed_data, data_item, caption_items):
        """Append caption supervision according to configured loading mode."""
        if self.caption_loading_mode == 'sample':
            processed_data.append(self._create_caption_sample_item(data_item))
            return

        for caption_item in caption_items:
            processed_data.append(self._create_data_item(data_item, caption_item))

    def _resolve_task_weights(self):
        """Resolve optional caption/qa task weights for caption+qa mode."""
        caption_weight = self.caption_task_weight
        qa_weight = self.qa_task_weight

        if caption_weight is None and qa_weight is None:
            return None, None

        caption_weight = 1.0 if caption_weight is None else float(caption_weight)
        qa_weight = 1.0 if qa_weight is None else float(qa_weight)

        if caption_weight < 0 or qa_weight < 0:
            raise ValueError("caption_task_weight and qa_task_weight must be non-negative")

        return caption_weight, qa_weight

    def _resample_items(self, items, target_count):
        """Resize supervision items list by sampling or repeating."""
        if target_count <= 0 or not items:
            return []

        if target_count == len(items):
            return list(items)

        if target_count < len(items):
            if self.stage == 'train':
                return random.sample(items, target_count)
            return list(items[:target_count])

        result = list(items)
        remaining = target_count - len(items)
        if self.stage == 'train':
            result.extend(random.choice(items) for _ in range(remaining))
        else:
            for idx in range(remaining):
                result.append(items[idx % len(items)])
        return result

    def _mix_caption_and_qa_items(self, data_item, caption_items, qa_items):
        """Mix caption and QA supervision with optional task weights."""
        caption_items = self._build_caption_supervision_items_for_mode(data_item, caption_items)
        caption_weight, qa_weight = self._resolve_task_weights()
        if caption_weight is None and qa_weight is None:
            return caption_items + qa_items

        if caption_weight == 0 and qa_weight == 0:
            return []

        if caption_weight == 0:
            return list(qa_items)
        if qa_weight == 0:
            return list(caption_items)

        total_source_count = len(caption_items) + len(qa_items)
        if total_source_count == 0:
            return []

        caption_ratio = caption_weight / (caption_weight + qa_weight)
        target_caption_count = round(total_source_count * caption_ratio)
        target_qa_count = total_source_count - target_caption_count

        if caption_items and target_caption_count == 0:
            target_caption_count = 1
            target_qa_count = max(0, total_source_count - target_caption_count)
        if qa_items and target_qa_count == 0:
            target_qa_count = 1
            target_caption_count = max(0, total_source_count - target_qa_count)

        mixed_items = []
        mixed_items.extend(self._resample_items(caption_items, target_caption_count))
        mixed_items.extend(self._resample_items(qa_items, target_qa_count))
        return mixed_items

    def _process_json_data(self, data):
        """Deprecated compatibility shim kept for external callers."""
        sample_dataset = WaveSampleDataset(data_root=self.data_root, split=self.split, opt=self.opt)
        return self._build_supervision_from_samples(sample_dataset.samples)

    def _build_supervision_from_samples(self, samples):
        processed_data = []
        for raw_item in samples:
            caption_items = self._build_caption_supervision(raw_item)
            qa_items = self._build_question_supervision(raw_item)

            if self.stage not in {'train', 'val', 'test'}:
                raise ValueError(f"Invalid stage: {self.stage}")

            if self.supervision_mode == 'caption':
                self._append_caption_items(processed_data, raw_item, caption_items)
            elif self.supervision_mode == 'qa':
                for qa_item in qa_items:
                    processed_data.append(self._create_data_item(raw_item, qa_item))
            elif self.supervision_mode == 'caption+qa':
                mixed_items = self._mix_caption_and_qa_items(raw_item, caption_items, qa_items)
                for supervision_item in mixed_items:
                    processed_data.append(self._create_data_item(raw_item, supervision_item))
            else:
                raise ValueError(f"Unsupported supervision_mode: {self.supervision_mode}")
        return processed_data

    def _build_supervision_index(self):
        """Build supervision items from sample-level records."""
        return self._build_supervision_from_samples(self.sample_dataset.samples)

    def _create_data_item(self, data_item, qa):
        """Create data items using NPZ file format (new data organization)."""
        radar_path = self.sample_dataset.get_radar_path(data_item)

        result = {
            'filename': radar_path,
            'question': qa.get('question') if qa else None,
            'answer': qa.get('answer') if qa else '',
            'caption': "\n".join(data_item['captions']),
            'task_type': qa.get('task_type', 'qa') if qa else 'qa',
        }

        # Add fileindex if available
        if 'fileindex' in data_item:
            result['fileindex'] = data_item['fileindex']

        radar_load_overrides = {
            key: data_item[key]
            for key in RADAR_LOAD_OVERRIDE_KEYS
            if key in data_item
        }
        nested_overrides = data_item.get('radar_load_overrides')
        if isinstance(nested_overrides, dict):
            radar_load_overrides.update({
                key: value
                for key, value in nested_overrides.items()
                if key in RADAR_LOAD_OVERRIDE_KEYS
            })
        if radar_load_overrides:
            result['radar_load_overrides'] = radar_load_overrides

        # Add qa_id if available (for QA data)
        if qa and 'qa_id' in qa:
            result['qa_id'] = qa['qa_id']

        return result

    def __len__(self):
        """Return number of utterances."""
        return len(self.data)

    def default_question(self):
        """Get a default question from predefined prompts.

        If use_random_question_for_caption is True, returns a random question.
        Otherwise, returns the first question in the list.
        """
        if self.use_random_question_for_caption:
            return random.choice(DEFAULT_QUESTION_PROMPTS)
        else:
            return DEFAULT_QUESTION_PROMPTS[0]

    @staticmethod
    def format_caption(question, answer):
        """Format question and answer into conversation format."""
        return [{
            "from": "human",
            "value": f"{DEFAULT_WAVE_INDICATOR}\n{question}"
        }, {
            "from": "gpt",
            "value": answer
        }]

    def __getitem__(self, index):
        """Get raw data item without preprocessing.

        Preprocessing is handled by datamodule.py's ConversationDatasetWrapper.
        This method only loads and returns raw data: question, answer, wave_embed.
        """
        instance = deepcopy(self.data[index])
        if instance['question'] is None:
            instance['question'] = self.default_question()

        # Load radar data from NPZ/NPY using the same pipeline as base_dataset.
        # Split records may attach loader overrides for domain-specific radar
        # preprocessing while reusing a shared encoder configuration.
        load_opt = deepcopy(self.opt)
        radar_load_overrides = instance.get('radar_load_overrides', {})
        if radar_load_overrides:
            for key, value in radar_load_overrides.items():
                if key not in RADAR_LOAD_OVERRIDE_KEYS:
                    raise KeyError(f"Unsupported radar load override key: {key}")
                load_opt[key] = value

        radar_data = load_radar_data(instance['filename'], load_opt)

        selected_views = self._selected_radar_views(load_opt)

        # On-the-fly static background overlay is only valid when all required
        # selected views are present in both motion and static data.  This keeps
        # uDoppler-only training from accidentally requiring range/azimuth.
        if self._static_views_cache is not None:
            if all(k in radar_data and k in self._static_views_cache for k in selected_views):
                if set(selected_views) == {'range_time', 'doppler_time', 'azimuth_time'}:
                    radar_data = mix_static_overlay(
                        self._static_views_cache,
                        radar_data,
                    )
                else:
                    # Single-view static overlay is intentionally not enabled by
                    # default; add a dedicated implementation if needed.
                    pass

        # Per-view random dropout (train-time regularization)
        for view_key, prob_key in [("range_time", "range_drop_prob"),
                                    ("doppler_time", "doppler_drop_prob"),
                                    ("azimuth_time", "azimuth_drop_prob")]:
            if view_key not in radar_data:
                continue
            prob = float(self.opt.get(prob_key, 0.0))
            if prob > 0 and random.random() < prob:
                radar_data[view_key] = np.zeros_like(radar_data[view_key])

        # Convert only configured views to tensors.
        wave_embed = {
            view_key: torch.tensor(radar_data[view_key]).float()
            for view_key in selected_views
            if view_key in radar_data
        }
        missing = [view_key for view_key in selected_views if view_key not in wave_embed]
        if missing:
            raise KeyError(
                f"Missing configured radar views {missing}; available keys: {list(radar_data.keys())}; "
                f"radar_views={self.opt.get('radar_views', 'all')}"
            )

        # Return raw data only - preprocessing done by datamodule.py
        result = {
            'filename': instance['filename'],
            'question': instance['question'],
            'answer': instance['answer'],
            'caption': instance['caption'],
            'wave_embed': wave_embed
        }

        # Add fileindex if available
        if 'fileindex' in instance:
            result['fileindex'] = instance['fileindex']

        # Add qa_id if available (for QA data)
        if 'qa_id' in instance:
            result['qa_id'] = instance['qa_id']

        return result


class WaveSampleDataset(Dataset):
    """Sample-level dataset that preserves raw split structure."""

    def __init__(self, data_root=None, split="train", opt=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.opt = opt or self._load_config(data_root)
        self.samples = self._load_samples()

    @staticmethod
    def _load_config(data_root):
        config_path = osp.join(data_root, 'radar_encoder_config.yaml')
        try:
            with open(config_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
        except yaml.constructor.ConstructorError:
            from yaml import UnsafeLoader
            with open(config_path, 'r') as file:
                yaml_data = yaml.load(file, Loader=UnsafeLoader)

        if yaml_data is None:
            raise ValueError(f"YAML config is empty: {config_path}")

        yaml_data = edict(yaml_data)
        if 'dataset_cfg' in yaml_data and yaml_data.dataset_cfg is not None:
            return edict(yaml_data.dataset_cfg)
        return edict(DEFAULT_REAL_CONFIG)

    def get_radar_path(self, data_item):
        """Get radar file path using the same logic as CLIP dataset.

        Supports:
        - generated/synthetic three-view NPZ files;
        - raw real cube NPY files when split records point directly to a raw
          real capture folder.
        """
        return resolve_radar_path(data_item, self.opt, require_exists=True)

    @staticmethod
    def _normalize_sample(raw_item):
        item = deepcopy(raw_item)
        if 'captions' not in item:
            item['captions'] = [item['classname']]
        return item

    def _load_split_file(self, split_file):
        if not osp.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            split_data = json.load(f)
        samples = [self._normalize_sample(sample) for sample in split_data.values()]
        if bool(self.opt.get('skip_missing_radar', False)):
            kept = []
            skipped = 0
            for sample in samples:
                radar_path = resolve_radar_path(sample, self.opt, require_exists=False)
                if osp.exists(radar_path):
                    kept.append(sample)
                else:
                    skipped += 1
            if skipped:
                log_message(
                    "LOAD",
                    f"Skipped {skipped} samples with missing radar files from {split_file}",
                    color="yellow",
                )
            samples = kept
        log_message("LOAD", f"Processed {len(samples)} samples from {split_file}", color="green")
        return samples

    def _load_samples(self):
        if self.split == 'real':
            raise ValueError("WaveSampleDataset does not support split='real'")

        if isinstance(self.split, list):
            samples = []
            for split_file in self.split:
                samples.extend(self._load_split_file(split_file))
            return samples

        if isinstance(self.split, str):
            return self._load_split_file(self.split)

        raise ValueError(f"Unsupported split specification: {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return deepcopy(self.samples[index])


# NOTE: mini_dataset() function has been removed as it was unused and called deleted preprocess().


def init_tokenizer_only(model_path):
    """Initialize only tokenizer without loading the full model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True  # Force offline mode
    )

    return tokenizer
