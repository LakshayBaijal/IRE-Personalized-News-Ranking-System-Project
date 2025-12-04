#!/bin/bash
# Quick start script for running the complete experiment

set -e

echo "=========================================="
echo "Personalized News Ranking Experiment"
echo "=========================================="

# Configuration
# Note: User simulation has ~9% engagement rate, need large query counts for meaningful data
NUM_BASELINE_QUERIES=50000      # Collect training data (~4,500 clicks expected)
NUM_PERSONALIZED_QUERIES=10000  # Test personalized model (~900 clicks expected)
NUM_AB_TEST_QUERIES=50000       # A/B test for statistical significance (~4,500 clicks expected)

# Step 1: Setup Elasticsearch and index articles
echo ""
echo "Step 1: Setting up Elasticsearch..."
echo "=========================================="
python main.py --mode setup

# Step 2: Run baseline system
echo ""
echo "Step 2: Running baseline system ($NUM_BASELINE_QUERIES queries)..."
echo "=========================================="
python main.py --mode baseline --num-queries $NUM_BASELINE_QUERIES

# Step 3: Train personalized model
echo ""
echo "Step 3: Training personalized model..."
echo "=========================================="
python main.py --mode train

# Step 4: Run personalized system
echo ""
echo "Step 4: Running personalized system ($NUM_PERSONALIZED_QUERIES queries)..."
echo "=========================================="
python main.py --mode personalized --num-queries $NUM_PERSONALIZED_QUERIES

# Step 5: Run A/B test
echo ""
echo "Step 5: Running A/B test ($NUM_AB_TEST_QUERIES queries)..."
echo "=========================================="
python main.py --mode ab-test --num-queries $NUM_AB_TEST_QUERIES

echo ""
echo "=========================================="
echo "✓ Experiment complete!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - data/logs/baseline_interactions.jsonl"
echo "  - data/logs/personalized_interactions.jsonl"
echo "  - data/logs/ab_test_results.json"
echo "  - models/personalized_model.json"
echo ""

