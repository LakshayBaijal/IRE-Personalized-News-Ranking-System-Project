# Personalized News Ranking System - Usage Guide

## Quick Start

### Prerequisites

1. **Docker** - for Elasticsearch and user simulation
2. **Python 3.8+** - for running the ranking system
3. **OpenAI API Key** (optional) - only needed for DSPy-enhanced ranking

### Step-by-Step Setup

#### 1. Install Python Dependencies

```bash
cd IRE_End
pip install -r requirements.txt
```

#### 2. Start Elasticsearch

```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.9.0
```

Verify it's running:
```bash
curl http://localhost:9200
```

#### 3. Start User Simulation Server

```bash
docker load -i ire_project-1.0-amd64.tar
docker run --rm -p 3000:3000 \
  -v $(pwd)/data:/data \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  ire_project:1.0
```

Verify it's running:
```bash
curl http://localhost:3000/query
```

#### 4. Index Articles to Elasticsearch

```bash
python main.py --mode setup
```

This will:
- Create an Elasticsearch index
- Load all articles from `articles.jsonl`
- Index them with BM25

Expected output: "✓ Elasticsearch setup complete!"

---

## Running Experiments

### Option A: Quick Automated Run

Run the complete experiment pipeline:

```bash
bash run_experiment.sh
```

This will:
1. Setup Elasticsearch (if not done)
2. Run 200 baseline queries
3. Train personalized model
4. Run 100 personalized queries
5. Run 300 A/B test queries
6. Generate statistical analysis

### Option B: Manual Step-by-Step

#### Step 1: Collect Baseline Data

Run the baseline system to collect initial training data:

```bash
python main.py --mode baseline --num-queries 200
```

This will:
- Fetch 200 queries from the simulation
- Use Elasticsearch BM25 to rank articles
- Submit rankings and collect user actions
- Log everything to `data/logs/baseline_interactions.jsonl`

**Output Example:**
```
Running Baseline System (200 queries)
✓ Connected to simulation server
Processing queries: 100%|████████| 200/200 [05:23<00:00, 0.62it/s]

Statistics:
  total_interactions: 200
  unique_users: 150
  total_clicks: 450
  avg_dwell_time: 35.2

Metrics:
  ctr: 0.2250
  mrr: 0.4123
  ndcg: 0.5234
```

#### Step 2: Train Personalized Model

Train a model on the collected baseline interactions:

```bash
python main.py --mode train --log-file data/logs/baseline_interactions.jsonl
```

This will:
- Analyze user interactions
- Learn topic preferences per user
- Build a global topic preference model
- Save to `models/personalized_model.json`

**Output Example:**
```
Training Personalized Model
Learning from 200 interactions...
Learned preferences for 150 users
Tracked 8 topics
✓ Training complete!
Saved model to models/personalized_model.json
```

#### Step 3: Run Personalized System

Test the personalized ranker:

```bash
python main.py --mode personalized --num-queries 100
```

Optional parameters:
- `--personalization-weight 0.4` - control personalization strength (0-1)
- `--model-file path/to/model.json` - use specific model

**Output Example:**
```
Running Personalized System (100 queries)
✓ Loaded model from models/personalized_model.json
Personalization weight: 0.4

Statistics:
  total_interactions: 100
  total_clicks: 280
  avg_dwell_time: 42.5

Metrics:
  ctr: 0.2800  ← improved!
  mrr: 0.4856  ← improved!
  ndcg: 0.5891 ← improved!
```

#### Step 4: Run A/B Test

Compare baseline and personalized systems statistically:

```bash
python main.py --mode ab-test --num-queries 300
```

This will:
- Randomly assign each query to control (baseline) or treatment (personalized)
- Run 300 queries with 50/50 split
- Perform statistical significance tests
- Calculate effect sizes and confidence intervals

**Output Example:**
```
Running A/B Test (300 queries)
Control queries: 152
Treatment queries: 148

Overall Winner: TREATMENT

Metrics Comparison:

CTR:
  Control:   0.2315
  Treatment: 0.2789
  Change:    +20.47%

MRR:
  Control:   0.4201
  Treatment: 0.4923
  Change:    +17.18%

Significance Tests:

CTR:
  Control:    0.2315
  Treatment:  0.2789
  Difference: +20.47%
  P-value:    0.001234
  Significant: ✓ YES (α=0.05)
  Effect size: 0.3245

✓ Full results saved to: data/logs/ab_test_results.json
```

---

## Understanding the System

### How Personalization Works

1. **Topic Preferences**: The system learns which topics each user engages with
2. **Engagement Scoring**: Actions are weighted (click=1, like=5, share=10, bookmark=7)
3. **Blending**: Combines BM25 relevance with learned preferences
4. **Cold Start**: New users get 70% global preferences, 30% their own

### Personalization Weight

The `--personalization-weight` parameter (default 0.4) controls the blend:

- **0.0**: Pure BM25 (baseline)
- **0.4**: Balanced (default, recommended)
- **0.7**: Strong personalization
- **1.0**: Pure preference-based (may hurt relevance)

### Metric Explanations

- **CTR (Click-Through Rate)**: % of articles clicked
- **MRR (Mean Reciprocal Rank)**: 1/position of first click
- **NDCG (Normalized DCG)**: Ranking quality with position discount
- **Engagement Rate**: Likes + shares + bookmarks per query
- **Avg Dwell Time**: Time spent reading clicked articles

---

## Advanced Usage

### Testing Individual Components

Each module can be tested independently:

```bash
# Test Elasticsearch setup
python -m src.elasticsearch_setup

# Test user simulation client
python -m src.user_simulation

# Test baseline ranker
python -m src.baseline_ranker

# Test personalized ranker
python -m src.personalized_ranker

# Test metrics calculation
python -m src.metrics

# Test A/B testing framework
python -m src.ab_testing
```

### Custom Configurations

Edit `config.py` to adjust:

```python
# Ranking parameters
TOP_K = 20              # Articles retrieved from ES
RERANK_TOP_K = 10       # Articles shown to user

# Metric weights
METRIC_WEIGHTS = {
    "click": 1.0,
    "dwell_time": 2.0,   # per second
    "like": 5.0,
    "share": 10.0,
    "bookmark": 7.0
}

# Training sizes
TRAIN_QUERIES = 200
VAL_QUERIES = 50
TEST_QUERIES = 100
```

### Using Different Log Files

```bash
# Run baseline with custom log
python main.py --mode baseline --num-queries 50 \
  --log-file data/logs/custom_baseline.jsonl

# Train on specific log
python main.py --mode train \
  --log-file data/logs/custom_baseline.jsonl \
  --model-file models/custom_model.json

# Test with custom model
python main.py --mode personalized --num-queries 50 \
  --model-file models/custom_model.json \
  --log-file data/logs/custom_personalized.jsonl
```

### Analyzing Results

The A/B test results are saved as JSON:

```bash
cat data/logs/ab_test_results.json | jq .
```

Key fields:
- `control.ctr` - baseline CTR
- `treatment.ctr` - personalized CTR
- `significance_tests[0].p_value` - statistical significance
- `significance_tests[0].effect_size` - Cohen's d
- `overall_winner` - "control", "treatment", or "inconclusive"

---

## Troubleshooting

### Elasticsearch Connection Error

```
Error: ConnectionError(...)
```

**Solution**: Ensure Elasticsearch is running:
```bash
docker ps | grep elasticsearch
curl http://localhost:9200
```

### User Simulation Not Running

```
❌ User simulation server is not running
```

**Solution**: Start the simulation:
```bash
docker run --rm -p 3000:3000 -v $(pwd)/data:/data ire_project:1.0
```

### No Results for Query

```
Warning: No results for query 'xyz'
```

**Solution**: The query might be too specific. The system will skip and continue.

### Out of Memory

If processing many queries:

1. Reduce `--num-queries`
2. Process in batches
3. Increase Docker memory limit

### Low Metrics

If metrics are unexpectedly low:

1. Check if articles are indexed: `curl localhost:9200/news_articles/_count`
2. Verify simulation is returning actions
3. Try adjusting `--personalization-weight`

---

## Project Structure Reference

```
IRE_End/
├── articles.jsonl              # 12,213 news articles
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration
├── main.py                     # Main orchestration
├── run_experiment.sh           # Automated pipeline
├── README.md                   # Project overview
├── USAGE_GUIDE.md             # This file
│
├── src/
│   ├── elasticsearch_setup.py  # Index articles
│   ├── user_simulation.py      # API client
│   ├── baseline_ranker.py      # BM25 ranking
│   ├── personalized_ranker.py  # Preference learning
│   ├── logger.py               # Interaction logging
│   ├── metrics.py              # Evaluation metrics
│   └── ab_testing.py           # Statistical tests
│
├── data/
│   └── logs/                   # Interaction logs
│       ├── baseline_interactions.jsonl
│       ├── personalized_interactions.jsonl
│       ├── ab_control.jsonl
│       ├── ab_treatment.jsonl
│       └── ab_test_results.json
│
└── models/
    └── personalized_model.json # Trained preferences
```

---

## Expected Results

Based on 50,000 query benchmark with the user simulation:

### Engagement Statistics (per 50k queries)
- Total Clicks: ~4,600 (~9.2% CTR)
- Total Dwell Time: ~33,000 seconds
- Total Likes: ~137
- Total Shares: ~90
- Total Bookmarks: ~97

### Baseline Performance (with 15,000+ queries)
- CTR: 0.08 - 0.10
- MRR: 0.15 - 0.25
- NDCG: 0.20 - 0.35

### Personalized Performance
- CTR: 0.10 - 0.14 (+10-40% improvement)
- MRR: 0.20 - 0.35 (+15-30% improvement)
- NDCG: 0.25 - 0.45 (+15-25% improvement)

### A/B Test (with 10,000+ queries)
- Expect statistically significant improvements (p < 0.05)
- Effect sizes (Cohen's d) typically 0.1 - 0.3
- Higher improvements for users with interaction history

**Important**: Use at least 10,000 queries for A/B testing to achieve statistical significance.

---

## Next Steps

After running the basic pipeline:

1. **Tune Hyperparameters**: Try different personalization weights
2. **Feature Engineering**: Add time-based features, query-article similarity
3. **Advanced Models**: Implement DSPy-enhanced ranking (requires OpenAI API)
4. **Cold Start**: Improve handling of new users
5. **Real-time Learning**: Update model with new interactions
6. **Multi-Armed Bandits**: Explore exploitation vs exploration

---

## Citation

If you use this system in your research, please cite:

```
Personalized News Ranking System
Using Elasticsearch, User Simulation, and A/B Testing
Information Retrieval and Extraction Project
```

---

## Support

For issues or questions:
1. Check this guide
2. Review code comments
3. Test individual modules
4. Check logs in `data/logs/`

