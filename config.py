"""Configuration settings for the personalized ranking system."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Elasticsearch configuration
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX_NAME = "news_articles"

# User simulation API configuration
USER_SIM_HOST = os.getenv("USER_SIM_HOST", "localhost")
USER_SIM_PORT = int(os.getenv("USER_SIM_PORT", 3000))
USER_SIM_BASE_URL = f"http://{USER_SIM_HOST}:{USER_SIM_PORT}"

# Articles file
ARTICLES_FILE = BASE_DIR / "articles.jsonl"

# Ranking configuration
TOP_K = 20  # Number of articles to retrieve and rank
RERANK_TOP_K = 10  # Number of articles to show to user

# ML Model configuration
RANKER_METHOD = os.getenv("RANKER_METHOD", "ensemble")  # 'cf', 'gbm', or 'ensemble'
CF_K_NEIGHBORS = 10  # Number of neighbors for collaborative filtering
GBM_NUM_ROUNDS = 100  # Number of boosting rounds for LightGBM

# Training configuration
# Note: User simulation has ~9% engagement rate, need large query counts
TRAIN_QUERIES = 50000   # Number of queries for training (baseline data collection)
VAL_QUERIES = 5000      # Number of queries for validation
TEST_QUERIES = 10000    # Number of queries for testing

# A/B Testing configuration
AB_TEST_ALPHA = 0.05  # Significance level
AB_TEST_MIN_SAMPLES = 500  # Minimum samples per variant (increased for statistical power)

# Metrics weights
METRIC_WEIGHTS = {
    "click": 1.0,
    "dwell_time": 2.0,  # Seconds to points conversion
    "like": 5.0,
    "share": 10.0,
    "bookmark": 7.0
}

