import os

# Set offline mode for Hugging Face to avoid network requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Import all components from the modularized files
from .clip_transformer import *
from .clip_text_encoder import *
from .clip_radar_encoder import *
from .clip_model import *