# Classical ML Methods for Personalized Ranking

This document describes the classical machine learning approaches used in this personalized ranking system.

## Overview

The system implements three well-established methods from recommendation systems and learning-to-rank literature:

1. **Collaborative Filtering (CF)**: Learn from user-item interaction patterns
2. **Learning to Rank with LightGBM**: Feature-based gradient boosting
3. **Ensemble**: Combine CF and LightGBM predictions

---

## Method 1: Collaborative Filtering

### Theory

Collaborative filtering is based on the assumption that users who agreed in the past will agree in the future. We use **user-based CF** with cosine similarity.

### Algorithm

1. **Build User-Item Matrix**:
   ```
   R[u, i] = engagement_score(user u, article i)
   ```

2. **Compute User Similarities**:
   ```
   similarity(u1, u2) = cosine(R[u1, :], R[u2, :])
   ```

3. **Predict Rating**:
   ```
   pred(u, i) = Σ(sim(u, v) × R[v, i]) / Σ|sim(u, v)|
                where v ∈ k-nearest neighbors of u
   ```

### Features

- **Engagement Scoring**: Weighted combination of actions
  - Click: 1.0
  - Like: 5.0
  - Share: 10.0
  - Bookmark: 7.0
  - Dwell time: 0.1 per second

- **Normalization**: Center ratings by user mean

- **Similarity Metric**: Cosine similarity (handles sparse data well)

- **K-Neighbors**: Default 10 (tunable)

### Advantages

- ✅ No feature engineering required
- ✅ Captures implicit user preferences
- ✅ Works well with sparse data
- ✅ Handles new items automatically

### Limitations

- ❌ Cold start for new users
- ❌ Scalability issues with large user base
- ❌ Doesn't use article content features

---

## Method 2: Learning to Rank (LightGBM)

### Theory

Learning to Rank (LTR) treats ranking as a supervised ML problem. We use **LambdaRank**, which optimizes for ranking metrics (NDCG) directly.

### Algorithm

LightGBM with LambdaRank objective:

1. **Feature Extraction**: Extract features for each (user, query, article) tuple

2. **Training**: Optimize pairwise ranking loss
   ```
   L = Σ loss(score(i), score(j)) × |ΔNDCG|
       where article i should rank higher than j
   ```

3. **Prediction**: Score = GBM(features)

### Features (8 dimensions)

1. **BM25 Score**: Baseline relevance from Elasticsearch
2. **User-Topic Affinity (Max)**: Highest affinity for article topics
3. **User-Topic Affinity (Mean)**: Average affinity for article topics
4. **Global Topic Popularity (Max)**: Most popular topic in article
5. **Global Topic Popularity (Mean)**: Average topic popularity
6. **Article Historical Engagement**: Past engagement with this article
7. **Number of Topics**: Topic count in article
8. **User Has Topic**: Binary flag if user has history with any topic

### Feature Computation

**User-Topic Affinity**:
```
affinity(u, t) = Σ engagement(u, articles with topic t)
                 normalized across all topics
```

**Global Topic Popularity**:
```
popularity(t) = Σ engagement(all users, articles with topic t)
                normalized across all topics
```

### Hyperparameters

- **Objective**: `lambdarank`
- **Metric**: `ndcg` @ [1, 3, 5, 10]
- **Learning Rate**: 0.05
- **Num Leaves**: 31
- **Boosting Rounds**: 100 (with early stopping)
- **Feature Fraction**: 0.8 (column sampling)
- **Bagging Fraction**: 0.8 (row sampling)

### Advantages

- ✅ Directly optimizes ranking metrics
- ✅ Handles heterogeneous features
- ✅ Provides feature importance
- ✅ Fast inference (<10ms per query)
- ✅ Robust to overfitting with regularization

### Limitations

- ❌ Requires training data
- ❌ Feature engineering needed
- ❌ May overfit to training distribution

---

## Method 3: Ensemble

### Theory

Ensemble methods combine multiple models to improve prediction quality and reduce variance.

### Algorithm

**Weighted Average**:
```
score_ensemble = α × score_CF + (1-α) × score_GBM
```

Default: α = 0.5 (equal weighting)

**Final Ranking**:
```
score_final = β × score_BM25 + (1-β) × score_ensemble
```

Default: β = 0.6 (60% relevance, 40% personalization)

### Advantages

- ✅ Combines strengths of both methods
- ✅ More robust than individual models
- ✅ Reduces variance
- ✅ Better generalization

### Limitations

- ❌ Slower inference (runs both models)
- ❌ More hyperparameters to tune

---

## Comparison of Methods

| Aspect | Collaborative Filtering | LightGBM | Ensemble |
|--------|------------------------|----------|----------|
| **Training Time** | Fast (~1s) | Medium (~30s) | Medium (~30s) |
| **Inference Time** | Fast (<5ms) | Fast (<5ms) | Medium (<10ms) |
| **Cold Start** | Poor | Good | Good |
| **Interpretability** | Medium | High | Medium |
| **Data Efficiency** | Good | Medium | Medium |
| **Scalability** | Medium | High | High |
| **Expected CTR Gain** | +10-15% | +15-20% | +20-30% |

---

## Implementation Details

### Collaborative Filtering

```python
class CollaborativeFilteringModel:
    def fit(self, interactions):
        # Build user-item matrix
        # Compute user similarities
        # Store top-k neighbors
    
    def predict(self, user_id, article_id):
        # Find k-nearest neighbors
        # Weighted average of neighbor ratings
        return predicted_score
```

**Key Functions**:
- `_compute_user_similarities()`: Cosine similarity between all users
- `predict()`: k-NN weighted prediction

### LightGBM Ranker

```python
class FeatureBasedRanker:
    def fit(self, interactions):
        # Extract features and labels
        # Train LightGBM with LambdaRank
    
    def predict(self, user_id, article):
        # Extract features
        # Run GBM inference
        return predicted_score
```

**Key Functions**:
- `_compute_statistics()`: Pre-compute user-topic affinities
- `_extract_article_features()`: Generate 8-dimensional feature vector
- `predict()`: LightGBM forward pass

### Ensemble Ranker

```python
class PersonalizedRanker:
    def __init__(self, method='ensemble'):
        self.cf_model = CollaborativeFilteringModel()
        self.gbm_model = FeatureBasedRanker()
        self.method = method
    
    def rerank(self, user_id, query, baseline_results):
        # Get CF predictions
        # Get GBM predictions
        # Combine with baseline
        return reranked_results
```

---

## Training Workflow

### 1. Data Preparation

```python
# Load logged interactions
interactions = load_logs('baseline_interactions.jsonl')

# Each interaction contains:
# - user_id, query_id, query_text
# - ranked_articles (with topics, BM25 scores)
# - actions (click, dwell, like, share, bookmark)
```

### 2. Model Training

```python
# Initialize ranker
ranker = PersonalizedRanker(method='ensemble')

# Train models
ranker.train(log_file)

# Save to disk
ranker.save('personalized_model.json')
```

### 3. Model Artifacts

Saved files:
- `personalized_model.json`: CF data + feature statistics
- `personalized_model_lgb.txt`: LightGBM booster
- `personalized_model_scaler.pkl`: StandardScaler for features

---

## Inference Workflow

### 1. Model Loading

```python
ranker = PersonalizedRanker(method='ensemble')
ranker.load('personalized_model.json')
```

### 2. Reranking

```python
# Get baseline results from Elasticsearch
baseline_results = es.search(query, top_k=20)

# Rerank with personalization
reranked = ranker.rerank(user_id, query, baseline_results, top_k=10)

# Return top-10 article IDs
article_ids = [r['uuid'] for r in reranked]
```

### 3. Score Computation

For each article:
1. Extract features
2. Compute CF score
3. Compute GBM score
4. Combine: `0.5 × CF + 0.5 × GBM`
5. Blend with BM25: `0.6 × BM25 + 0.4 × personalized`
6. Sort by final score

---

## Hyperparameter Tuning

### Collaborative Filtering

- **k_neighbors**: [5, 10, 15, 20]
  - Lower: More personalized but noisy
  - Higher: More stable but less personalized

### LightGBM

- **learning_rate**: [0.01, 0.05, 0.1]
- **num_leaves**: [15, 31, 63]
- **min_data_in_leaf**: [10, 20, 50]
- **feature_fraction**: [0.6, 0.8, 1.0]

### Ensemble

- **cf_weight**: [0.3, 0.5, 0.7]
  - Lower: More reliance on LightGBM
  - Higher: More reliance on CF

- **personalization_weight**: [0.2, 0.4, 0.6]
  - Lower: More reliance on BM25 (safe)
  - Higher: More personalization (risky)

---

## Evaluation

### Offline Metrics

Computed on held-out test set:
- **NDCG@10**: Ranking quality
- **MRR**: First relevant item position
- **CTR**: Click-through rate
- **Engagement Rate**: Likes + shares + bookmarks

### Online Metrics (A/B Test)

Statistical tests comparing control vs treatment:
- **Two-sample t-tests** on each metric
- **P-values** and **confidence intervals**
- **Effect sizes** (Cohen's d)

### Expected Results

| Method | CTR | MRR | NDCG | Engagement |
|--------|-----|-----|------|------------|
| Baseline (BM25) | 0.22 | 0.41 | 0.52 | 0.15 |
| CF | 0.25 | 0.45 | 0.56 | 0.18 |
| LightGBM | 0.27 | 0.47 | 0.58 | 0.20 |
| Ensemble | 0.29 | 0.49 | 0.61 | 0.22 |

---

## References

### Collaborative Filtering

- Sarwar et al. (2001). "Item-based collaborative filtering recommendation algorithms"
- Herlocker et al. (2004). "Evaluating collaborative filtering recommender systems"

### Learning to Rank

- Burges (2010). "From RankNet to LambdaRank to LambdaMART: An Overview"
- Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

### Ensemble Methods

- Dietterich (2000). "Ensemble methods in machine learning"
- Jahrer et al. (2010). "Combining predictions for accurate recommender systems"

---

## Implementation Tips

### 1. Cold Start Handling

For new users with no history:
- CF: Use global average or most popular items
- LightGBM: Rely on global features (topic popularity)
- Ensemble: Falls back to BM25-heavy ranking

### 2. Scalability

For production deployment:
- **CF**: Use approximate nearest neighbors (Annoy, FAISS)
- **LightGBM**: Cache predictions for popular items
- **Batch Processing**: Precompute features offline

### 3. Online Learning

Update models periodically:
- Incremental learning for CF (update matrix)
- Retrain LightGBM weekly/monthly
- A/B test before deploying updates

### 4. Monitoring

Track in production:
- Model inference latency
- Feature drift (distribution changes)
- Metric degradation
- Edge cases (empty results, errors)

---

## Conclusion

This implementation combines classical, well-understood ML methods that have proven effective in real-world recommendation systems. The ensemble approach balances the strengths of collaborative filtering (user similarity) and learning-to-rank (feature-based optimization) to achieve strong personalization with minimal complexity.

