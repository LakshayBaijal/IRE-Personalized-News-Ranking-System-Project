# Personalized News Ranking System

An end-to-end personalized ranking system for news articles using Elasticsearch, classical Machine Learning (Collaborative Filtering + LightGBM), and A/B testing. This project demonstrates a complete pipeline for personalized news retrieval, from data indexing to statistical evaluation.


## Project Execution Video
[Screencast from 2025-12-30 23-13-54.webm](https://github.com/user-attachments/assets/ac11a6a9-2c76-4733-8c12-5261aba8beda)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a personalized news ranking system that adapts to individual user preferences. It uses **Elasticsearch** for efficient baseline retrieval and re-ranks results using a **hybrid personalization model**. The system's effectiveness is validated through a simulated environment and rigorous **A/B testing**.

## Key Features

- **Baseline Retrieval**: BM25-based search using Elasticsearch.
- **Hybrid Personalization**: Combines Collaborative Filtering (user-user similarity) and Learning to Rank (LightGBM).
- **User Simulation**: A Dockerized simulation server that models realistic user interactions (clicks, likes, shares, bookmarks).
- **Comprehensive Logging**: detailed tracking of all user interactions for offline analysis.
- **A/B Testing Framework**: Statistical significance testing (t-tests, Cohen's d) to compare personalized ranking against the baseline.
- **Scalable Design**: Efficient batch processing and modular architecture.

## 🏗 System Architecture

The system consists of a Python ranking backend interacting with an Elasticsearch cluster and a User Simulation API.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Simulation API                     │
│              (Docker container on port 3000)                │
└──────────────┬─────────────────────────┬────────────────────┘
               │ /query                  │ /ranklist
               │ (get user & query)      │ (submit ranking)
               │                         │
┌──────────────▼─────────────────────────▼────────────────────┐
│                   Ranking System (Python)                    │
│                                                              │
│  ┌────────────────┐    ┌──────────────────┐                │
│  │    Baseline    │    │   Personalized   │                │
│  │    Ranker      │    │     Ranker       │                │
│  │  (BM25 only)   │    │  (BM25 + prefs)  │                │
│  └────────┬───────┘    └────────┬─────────┘                │
│           │                     │                           │
│         ┌─▼─────────────────────▼──┐                        │
│         │   Interaction Logger     │                        │
│         └─┬─────────────────────┬──┘                        │
│           │                     │                           │
│         ┌─▼─────────────────────▼──┐                        │
│         │   Metrics & A/B Test     │                        │
│         └──────────────────────────┘                        │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼────────────────────┐
│      Elasticsearch Index          │
│      (12,213 articles)            │
└───────────────────────────────────┘
```

## Methodology

The personalization engine employs a hybrid approach:

1.  **Collaborative Filtering (CF)**:
    *   **Concept**: Users who agreed in the past will agree in the future.
    *   **Implementation**: User-based CF using cosine similarity on engagement vectors.
    *   **Pros**: Captures implicit user groups and trends.

2.  **Learning to Rank (LightGBM)**:
    *   **Concept**: Treat ranking as a supervised learning problem (LambdaRank).
    *   **Features**: 8 dimensions including User-Topic Affinity, Global Topic Popularity, and Article Engagement.
    *   **Pros**: Directly optimizes ranking metrics (NDCG).

3.  **Ensemble**:
    *   **Implementation**: A weighted average of CF and LightGBM scores, blended with the original BM25 relevance score.
    *   **Equation**: `Final_Score = 0.6 * BM25 + 0.4 * (0.5 * CF + 0.5 * LightGBM)`

## Installation & Setup

### Prerequisites
- **Docker** (for Elasticsearch and User Simulation)
- **Python 3.8+**

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Elasticsearch
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.12.0
```

### Step 3: Start User Simulation
Load the provided Docker image and run the simulation server.
```bash
docker load -i ire_project-1.0-amd64.tar
docker run --rm -p 3000:3000 \
  -v $(pwd)/data:/data \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  ire_project:1.0
```

### Step 4: Index Articles
Initialize the Elasticsearch index and load the news articles.
```bash
python main.py --mode setup
```
*Expected Output:* `✓ Elasticsearch setup complete!`

## 💻 Usage

You can run the full experiment pipeline automatically or execute individual steps.

### Automated Run
Run the complete pipeline (Baseline -> Train -> Personalized -> A/B Test):
```bash
bash run_experiment.sh
```

### Manual Steps

**1. Run Baseline System**
Collect initial interaction data using standard BM25 ranking.
```bash
python main.py --mode baseline --num-queries 200
```

**2. Train Personalized Model**
Train the personalization models using the collected logs.
```bash
python main.py --mode train --log-file data/logs/baseline_interactions.jsonl
```

**3. Run Personalized System**
Evaluate the personalized ranker.
```bash
python main.py --mode personalized --num-queries 100
```

**4. Run A/B Test**
Compare the Baseline (Control) vs. Personalized (Treatment) systems.
```bash
python main.py --mode ab-test --num-queries 500
```

## Results

The system evaluates performance using **CTR (Click-Through Rate)**, **MRR (Mean Reciprocal Rank)**, **NDCG**, and **Engagement Rate**.

### Benchmark Performance
*Based on 50,000 query benchmark.*

| Metric | Baseline (BM25) | Personalized | Improvement |
|--------|----------------|--------------|-------------|
| **CTR** | 0.22 | **0.28** | **+27%** |
| **MRR** | 0.41 | **0.49** | **+20%** |
| **NDCG** | 0.52 | **0.59** | **+13%** |
| **Engagement** | 0.15 | **0.22** | **+47%** |

### A/B Testing Results
The A/B test typically shows statistically significant improvements for the Personalized system (Treatment) over the Baseline (Control).

*Example Output:*
```
Overall Winner: TREATMENT

Metrics Comparison:
CTR:        +20.47% (p < 0.05)
MRR:        +17.18% (p < 0.05)
NDCG:       +12.55% (p < 0.05)
```

*Note: Actual results depend on the number of queries run. A minimum of 10,000 queries is recommended for statistical significance due to the cold-start nature of the simulation.*

## Project Structure

```
IRE_End/
├── articles.jsonl              # Dataset of 12,213 news articles
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings (weights, paths)
├── main.py                     # Main CLI entry point
├── run_experiment.sh           # Automated experiment script
├── src/
│   ├── elasticsearch_setup.py  # Indexing logic
│   ├── user_simulation.py      # API Client for simulation
│   ├── baseline_ranker.py      # BM25 implementation
│   ├── personalized_ranker.py  # ML models (CF, LightGBM)
│   ├── metrics.py              # Calculation of CTR, NDCG, etc.
│   ├── ab_testing.py           # Statistical analysis
│   └── logger.py               # Interaction logging
└── data/
    └── logs/                   # Stored interaction logs
```

Keywords- 
tutorial, from-scratch, how-to, example, demo, install, run, dataset, colab, c++, linux, system-calls, object-detection, opencv
