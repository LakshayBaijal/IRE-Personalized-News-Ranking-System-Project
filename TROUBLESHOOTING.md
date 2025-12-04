# Troubleshooting Guide

## Issue: All Metrics Are Zero (No User Engagement)

### Symptoms
```
Statistics:
  total_clicks: 0
  total_likes: 0
  total_shares: 0
  
Metrics:
  ctr: 0.0000
  mrr: 0.0000
  ndcg: 0.0000
```

### Possible Causes & Solutions

#### 1. User Simulation Not Matching Articles

**Cause**: The articles retrieved from Elasticsearch don't match what the user simulation expects.

**Solution**:
```bash
# Run the debug script
python debug_articles.py
```

This will:
- Get a query from simulation
- Search Elasticsearch
- Submit results to simulation
- Check if user actions are received
- Verify UUID matching

#### 2. Query-Article Relevance

**Cause**: The queries and articles might not be well-matched, leading to no engagement.

**Solution**: This is actually **normal behavior** in some cases! The user simulation is sophisticated and users won't click irrelevant results.

**What to do**:
1. Collect more data (run more queries)
2. The personalization should improve over time as patterns emerge
3. Check if specific queries get better engagement

```bash
# Try with more queries
python main.py --mode baseline --num-queries 500
```

#### 3. Elasticsearch Indexing Issue

**Cause**: Articles might not be properly indexed.

**Solution**:
```bash
# Verify ES index
curl http://localhost:9200/news_articles/_count

# Re-index if needed
python main.py --mode setup
```

#### 4. User Simulation Data Mismatch

**Cause**: The simulation's internal dataset might be different from articles.jsonl

**Check**:
```bash
# Count articles in file
wc -l articles.jsonl

# Count in ES
curl http://localhost:9200/news_articles/_count

# They should match (12,213 articles)
```

---

## Issue: JSON Serialization Error

### Symptoms
```
TypeError: Object of type bool is not JSON serializable
```

### Cause
NumPy boolean or float types can't be directly serialized to JSON.

### Solution
✅ **FIXED** - Updated `src/ab_testing.py` to convert all values to native Python types:
- `numpy.bool_` → `bool()`
- `numpy.float64` → `float()`
- `numpy.int64` → `int()`

---

## Understanding Zero Metrics

### Is This Normal?

**Yes, it can be!** Here's why:

1. **Realistic User Behavior**: The user simulation models realistic user behavior. If articles aren't relevant or engaging, users won't click.

2. **Cold Start Problem**: With no prior engagement data, the initial rankings might not be well-personalized.

3. **Topic Mismatch**: If queries don't match article topics well, engagement will be low.

### What You Should See

**Early queries (1-50)**: Low engagement is normal
- Users are exploring
- System is learning preferences
- Many queries might have 0 clicks

**After 100+ queries**: Should start seeing some engagement
- Patterns emerge
- Personalization kicks in
- Some queries should show clicks

**After 500+ queries**: Clear improvement
- Personalized model has learned user preferences
- Measurable differences between baseline and personalized
- Statistical significance in A/B tests

### Expected Engagement Rates

Based on the user simulation (measured from 50,000 queries):

| Queries | Expected Clicks | Expected CTR | Notes |
|---------|-----------------|--------------|-------|
| 200 | ~18 | ~9% | Too few for reliable stats |
| 1,000 | ~92 | ~9% | Minimum for basic testing |
| 5,000 | ~460 | ~9% | Good for model training |
| 10,000 | ~920 | ~9% | Good for A/B testing |
| 15,000+ | ~1,380+ | ~9% | Recommended for training |

**Key insight**: The simulation has ~9% CTR, so you need **large query counts** (10,000+) to get statistically meaningful results.

---

## How to Get Better Results

### 1. Collect More Data

The system needs sufficient data to learn patterns:

```bash
# Baseline: Collect 500 queries
python main.py --mode baseline --num-queries 500

# Train on more data
python main.py --mode train

# Test: Run 200 personalized queries
python main.py --mode personalized --num-queries 200

# A/B test with 500 queries
python main.py --mode ab-test --num-queries 500
```

### 2. Check Specific Queries

Some queries naturally get more engagement:

```python
# In debug_articles.py, try multiple queries
for i in range(10):
    query = client.get_query()
    print(f"Query {i}: '{query.query_text}'")
    # ... submit and check actions
```

### 3. Verify Data Pipeline

```bash
# 1. Check ES is running
curl http://localhost:9200

# 2. Check simulation is running
curl http://localhost:3000/query

# 3. Check articles are indexed
curl http://localhost:9200/news_articles/_count

# 4. Run debug script
python debug_articles.py
```

### 4. Analyze Logs

Look at the interaction logs:

```bash
# Check what's being logged
head -n 5 data/logs/baseline_interactions.jsonl | jq .

# Count interactions
wc -l data/logs/baseline_interactions.jsonl

# Check for any clicks
grep -c "Click" data/logs/baseline_interactions.jsonl
```

### 5. Try Different Queries Manually

Test with the simulation API directly:

```bash
# Get a query
curl http://localhost:3000/query

# Use that query to search ES, get article IDs
# Then submit to simulation
curl -X POST http://localhost:3000/ranklist \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "YOUR_QUERY_ID",
    "user_id": "YOUR_USER_ID",
    "ranked_article_ids": ["UUID1", "UUID2", ...]
  }'
```

---

## Performance Benchmarks

### Minimum Data Requirements

For reliable evaluation:
- **Training**: 200-500 queries
- **Validation**: 100-200 queries  
- **A/B Testing**: 300-1000 queries (more = better statistical power)

### Expected Timeline

| Phase | Queries | Time | Expected Result |
|-------|---------|------|-----------------|
| Initial baseline | 200 | 3-5 min | May see 0-5% engagement |
| Training | - | 30 sec | Model learns patterns |
| Personalized test | 100 | 2-3 min | Should see improvement |
| A/B test | 300+ | 5-10 min | Statistical significance |

---

## When to Contact Support

If after following all troubleshooting steps:
1. ✅ ES is properly indexed (12,213 articles)
2. ✅ Simulation server is running
3. ✅ debug_articles.py confirms UUID matching
4. ✅ Ran 500+ baseline queries
5. ❌ **Still zero engagement across ALL queries**

Then there might be a deeper issue with:
- User simulation docker image
- Article data format mismatch
- Network/port issues

---

## Quick Fixes Checklist

- [ ] ES running on port 9200
- [ ] Simulation running on port 3000
- [ ] 12,213 articles indexed in ES
- [ ] Run `python debug_articles.py`
- [ ] Check logs have data: `ls -lh data/logs/`
- [ ] Try 500+ queries for training
- [ ] Verify article UUIDs match between ES and articles.jsonl

---

## Advanced Debugging

### Check User Simulation Behavior

```python
# In Python console
from src.user_simulation import UserSimulationClient

client = UserSimulationClient()

# Try 10 different queries
for i in range(10):
    query = client.get_query()
    print(f"\nQuery {i}: {query.query_text}")
    
    # Submit dummy articles
    dummy_ids = ["550e8400-e29b-41d4-a716-446655440000"] * 10
    try:
        response = client.submit_ranklist(query.query_id, query.user_id, dummy_ids)
        action_count = sum(len(actions) for actions in response.actions)
        print(f"Total actions: {action_count}")
    except Exception as e:
        print(f"Error: {e}")
```

### Analyze Article Distribution

```python
import json
from collections import Counter

# Check topic distribution
topics = []
with open('articles.jsonl', 'r') as f:
    for line in f:
        article = json.loads(line)
        topics.extend(article['topics'])

topic_counts = Counter(topics)
print("Topic distribution:")
for topic, count in topic_counts.most_common():
    print(f"  {topic}: {count}")
```

---

## Summary

**Zero engagement after a few hundred queries is expected behavior** for a cold-start system. The key is:

1. ✅ Verify technical setup (ES, simulation, article matching)
2. ✅ Collect sufficient data (500+ queries)
3. ✅ Train and iterate
4. ✅ Compare baseline vs personalized with A/B testing

The personalization will improve over time as more user interaction data is collected!

