# Personalized News Ranking System

An end-to-end personalized ranking system for news articles using Elasticsearch, classical ML (Collaborative Filtering + LightGBM), and A/B testing.

## Project Structure

```
IRE_End/
├── articles.jsonl              # News articles dataset
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── src/
│   ├── __init__.py
│   ├── elasticsearch_setup.py  # Elasticsearch indexing
│   ├── user_simulation.py      # API client for user simulation
│   ├── baseline_ranker.py      # Baseline Elasticsearch ranking
│   ├── logger.py               # Interaction logging system
│   ├── personalized_ranker.py  # DSPy-based personalized ranking
│   ├── metrics.py              # Evaluation metrics
│   └── ab_testing.py           # A/B testing framework
├── data/
│   └── logs/                   # User interaction logs
├── models/                     # Trained models
└── main.py                     # Main orchestration script
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Elasticsearch (Docker):
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.12.0
```

3. Load and run user simulation:
```bash
docker load -i ire_project-1.0-amd64.tar
docker run --rm -p 3000:3000 \
  -v $(pwd)/data:/data \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  ire_project:1.0
```

## Usage

### 1. Index Articles to Elasticsearch
```bash
python -m src.elasticsearch_setup
```

### 2. Run Baseline System
```bash
python main.py --mode baseline --num-queries 100
```

### 3. Train Personalized Ranker
```bash
# Train with ensemble method (recommended)
python main.py --mode train --method ensemble

# Or train with specific method
python main.py --mode train --method cf      # Collaborative Filtering only
python main.py --mode train --method gbm     # Gradient Boosting only
```

### 4. Run Personalized System
```bash
python main.py --mode personalized --num-queries 100 --method ensemble
```

### 5. A/B Testing
```bash
python main.py --mode ab-test --num-queries 500
```

## Metrics

- **Click-Through Rate (CTR)**: Percentage of clicked articles
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first click
- **Normalized Discounted Cumulative Gain (NDCG)**: Weighted ranking quality
- **Dwell Time**: Average time spent on articles
- **Engagement Rate**: Likes, shares, bookmarks per query

## Approach

1. **Baseline**: Elasticsearch BM25 ranking
2. **Feature Extraction**: Topics + learned user preferences + engagement history
3. **Personalization**: Classical ML approaches
   - **Collaborative Filtering**: User-user similarity with matrix factorization
   - **Learning to Rank**: LightGBM with LambdaRank objective
   - **Ensemble**: Weighted combination of CF + LightGBM
4. **Evaluation**: Statistical significance testing with A/B framework

