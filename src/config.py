import os
import torch
import logging

# Project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EMBEDDING_DIR = os.path.join(DATA_DIR, 'embeddings')

# Data files
TRAIN_DATA = os.path.join(PROCESSED_DATA_DIR, 'train.pkl')
VAL_DATA = os.path.join(PROCESSED_DATA_DIR, 'val.pkl')
TEST_DATA = os.path.join(PROCESSED_DATA_DIR, 'test.pkl')
PREPROCESSED_DATA = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_data.pkl')

# Embedding files
GLOVE = os.path.join(EMBEDDING_DIR, 'glove.6B.300d.txt')
WORD2VEC = os.path.join(EMBEDDING_DIR, 'GoogleNews-vectors-negative300.bin')
FASTTEXT = os.path.join(EMBEDDING_DIR, 'crawl-300d-2M-subword.vec')

# Model configuration
SEARCH_SPACE_FILE = os.path.join(BASE_DIR, 'src', 'nas', 'search_space.json')
MAX_SEQ_LENGTH = 128  # This can be set from the search space as needed

# NAS configuration
EXPERIMENT_NAME = 'nas_ner_experiment'
TRIAL_CONCURRENCY = 1
MAX_TRIAL_NUMBER = 200
MAX_EXPERIMENT_DURATION = '3d'

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(BASE_DIR, 'nas_ner.log')

# Create necessary directories
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=LOG_FILE)

# NAS configuration
NAS_CONFIG = {
    'experiment_name': EXPERIMENT_NAME,
    'trial_concurrency': TRIAL_CONCURRENCY,
    'max_trial_number': MAX_TRIAL_NUMBER,
    'max_experiment_duration': MAX_EXPERIMENT_DURATION,
    'search_space_file': SEARCH_SPACE_FILE,
    'training_service': {
        'platform': 'local'
    },
    'trial_code_directory': BASE_DIR,
    'batch_size': 32
}