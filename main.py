"""Main orchestration script for personalized ranking system."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random
import config
from src.elasticsearch_setup import ElasticsearchSetup
from src.user_simulation import UserSimulationClient
from src.baseline_ranker import BaselineRanker
from src.personalized_ranker import PersonalizedRanker
from src.logger import InteractionLogger
from src.metrics import MetricsCalculator
from src.ab_testing import ABTest


def setup_elasticsearch():
    """Set up and index articles in Elasticsearch."""
    print("=" * 60)
    print("Setting up Elasticsearch")
    print("=" * 60)
    
    es_setup = ElasticsearchSetup()
    es_setup.setup()


def run_baseline(num_queries: int = 100, log_file: Path = None):
    """Run baseline ranking system."""
    print("=" * 60)
    print(f"Running Baseline System ({num_queries} queries)")
    print("=" * 60)
    
    # Initialize components
    client = UserSimulationClient()
    ranker = BaselineRanker()
    
    if log_file is None:
        log_file = config.LOGS_DIR / "baseline_interactions.jsonl"
    
    logger = InteractionLogger(log_file)
    
    # Check if simulation server is running
    if not client.health_check():
        print(f"\n❌ User simulation server is not running at {client.base_url}")
        print("Please start it with:")
        print("  docker run --rm -p 3000:3000 -v $(pwd)/data:/data ire_project:1.0")
        return
    
    print(f"✓ Connected to simulation server at {client.base_url}")
    print(f"Logging to: {log_file}")
    
    # Cumulative tracking for long runs
    cumulative_clicks = 0
    cumulative_dwell = 0.0
    cumulative_likes = 0
    cumulative_shares = 0
    cumulative_bookmarks = 0
    
    # Print progress every N queries for long runs
    progress_interval = max(1000, num_queries // 10)
    
    # Run queries
    for i in tqdm(range(num_queries), desc="Processing queries"):
        try:
            # Get query from simulation
            query = client.get_query()
            
            # Get baseline ranking
            ranked_ids = ranker.get_ranked_ids(query.query_text, top_k=config.RERANK_TOP_K)
            
            if not ranked_ids:
                print(f"\nWarning: No results for query '{query.query_text}'")
                continue
            
            # Submit to simulation and get actions
            response = client.submit_ranklist(query.query_id, query.user_id, ranked_ids)
            
            # Get full article data for logging
            ranked_articles = []
            for article_id in ranked_ids:
                article = ranker.get_article_by_id(article_id)
                if article:
                    article["uuid"] = article_id
                    ranked_articles.append(article)
            
            # Log interaction
            logger.log_interaction(
                user_id=query.user_id,
                query_id=query.query_id,
                query_text=query.query_text,
                ranked_articles=ranked_articles,
                actions=[[a.__dict__ for a in action_list] for action_list in response.actions],
                ranker_type="baseline"
            )
            
            # Track cumulative stats
            for action_list in response.actions:
                for action in action_list:
                    if action.action_type == "Click":
                        cumulative_clicks += 1
                    elif action.action_type == "Dwell" and action.dwell_time:
                        cumulative_dwell += action.dwell_time
                    elif action.action_type == "Like":
                        cumulative_likes += 1
                    elif action.action_type == "Share":
                        cumulative_shares += 1
                    elif action.action_type == "Bookmark":
                        cumulative_bookmarks += 1
            
            # Print periodic summary for long runs
            if (i + 1) % progress_interval == 0 and num_queries > 1000:
                tqdm.write(f"\n--- Progress Update ({i+1}/{num_queries} queries) ---")
                tqdm.write(f"  Clicks: {cumulative_clicks} | Dwell: {cumulative_dwell:.0f}s | "
                          f"Likes: {cumulative_likes} | Shares: {cumulative_shares} | Bookmarks: {cumulative_bookmarks}")
            
        except Exception as e:
            print(f"\nError processing query {i+1}: {e}")
            continue
    
    # Print cumulative summary
    print("\n" + "=" * 60)
    print("Cumulative User Interaction Summary")
    print("=" * 60)
    print(f"  Total Clicks: {cumulative_clicks}")
    print(f"  Total Dwell Time: {cumulative_dwell:.0f} seconds")
    print(f"  Total Likes: {cumulative_likes}")
    print(f"  Total Shares: {cumulative_shares}")
    print(f"  Total Bookmarks: {cumulative_bookmarks}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Baseline Run Complete")
    print("=" * 60)
    stats = logger.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Calculate and display metrics
    interactions = InteractionLogger.load_logs(log_file)
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(interactions)
    
    print("\nMetrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    return log_file


def train_personalized_model(
    log_file: Path,
    output_model: Path = None,
    method: str = 'ensemble'
):
    """Train personalized ranking model from logs."""
    print("=" * 60)
    print("Training Personalized Model")
    print("=" * 60)
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print("Please run baseline mode first to collect training data.")
        return
    
    if output_model is None:
        output_model = config.MODELS_DIR / "personalized_model.json"
    
    print(f"Training from: {log_file}")
    print(f"Output model: {output_model}")
    print(f"Method: {method}")
    
    # Train model
    ranker = PersonalizedRanker(method=method)
    ranker.train(log_file)
    
    # Save model
    ranker.save(output_model)
    
    print("\n✓ Training complete!")
    return output_model


def run_personalized(
    num_queries: int = 100,
    model_file: Path = None,
    log_file: Path = None,
    method: str = 'ensemble'
):
    """Run personalized ranking system."""
    print("=" * 60)
    print(f"Running Personalized System ({num_queries} queries)")
    print("=" * 60)
    
    if model_file is None:
        model_file = config.MODELS_DIR / "personalized_model.json"
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        print("Please train a model first using --mode train")
        return
    
    # Initialize components
    client = UserSimulationClient()
    ranker = PersonalizedRanker(method=method)
    ranker.load(model_file)
    
    if log_file is None:
        log_file = config.LOGS_DIR / "personalized_interactions.jsonl"
    
    logger = InteractionLogger(log_file)
    
    # Check if simulation server is running
    if not client.health_check():
        print(f"\n❌ User simulation server is not running at {client.base_url}")
        return
    
    print(f"✓ Connected to simulation server")
    print(f"✓ Loaded model from {model_file}")
    print(f"Method: {ranker.method}")
    print(f"Logging to: {log_file}")
    
    # Cumulative tracking for long runs
    cumulative_clicks = 0
    cumulative_dwell = 0.0
    cumulative_likes = 0
    cumulative_shares = 0
    cumulative_bookmarks = 0
    progress_interval = max(1000, num_queries // 10)
    
    # Run queries
    for i in tqdm(range(num_queries), desc="Processing queries"):
        try:
            # Get query from simulation
            query = client.get_query()
            
            # Get personalized ranking
            ranked_ids = ranker.get_ranked_ids(
                query.user_id,
                query.query_text,
                top_k=config.RERANK_TOP_K
            )
            
            if not ranked_ids:
                print(f"\nWarning: No results for query '{query.query_text}'")
                continue
            
            # Submit to simulation and get actions
            response = client.submit_ranklist(query.query_id, query.user_id, ranked_ids)
            
            # Get full article data for logging
            ranked_articles = []
            for article_id in ranked_ids:
                article = ranker.baseline_ranker.get_article_by_id(article_id)
                if article:
                    article["uuid"] = article_id
                    ranked_articles.append(article)
            
            # Log interaction
            logger.log_interaction(
                user_id=query.user_id,
                query_id=query.query_id,
                query_text=query.query_text,
                ranked_articles=ranked_articles,
                actions=[[a.__dict__ for a in action_list] for action_list in response.actions],
                ranker_type="personalized",
                metadata={"method": ranker.method}
            )
            
            # Track cumulative stats
            for action_list in response.actions:
                for action in action_list:
                    if action.action_type == "Click":
                        cumulative_clicks += 1
                    elif action.action_type == "Dwell" and action.dwell_time:
                        cumulative_dwell += action.dwell_time
                    elif action.action_type == "Like":
                        cumulative_likes += 1
                    elif action.action_type == "Share":
                        cumulative_shares += 1
                    elif action.action_type == "Bookmark":
                        cumulative_bookmarks += 1
            
            # Print periodic summary for long runs
            if (i + 1) % progress_interval == 0 and num_queries > 1000:
                tqdm.write(f"\n--- Progress Update ({i+1}/{num_queries} queries) ---")
                tqdm.write(f"  Clicks: {cumulative_clicks} | Dwell: {cumulative_dwell:.0f}s | "
                          f"Likes: {cumulative_likes} | Shares: {cumulative_shares} | Bookmarks: {cumulative_bookmarks}")
            
        except Exception as e:
            print(f"\nError processing query {i+1}: {e}")
            continue
    
    # Print cumulative summary
    print("\n" + "=" * 60)
    print("Cumulative User Interaction Summary")
    print("=" * 60)
    print(f"  Total Clicks: {cumulative_clicks}")
    print(f"  Total Dwell Time: {cumulative_dwell:.0f} seconds")
    print(f"  Total Likes: {cumulative_likes}")
    print(f"  Total Shares: {cumulative_shares}")
    print(f"  Total Bookmarks: {cumulative_bookmarks}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Personalized Run Complete")
    print("=" * 60)
    stats = logger.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Calculate and display metrics
    interactions = InteractionLogger.load_logs(log_file)
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(interactions)
    
    print("\nMetrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    return log_file


def run_ab_test(
    num_queries: int = 200,
    model_file: Path = None,
    method: str = 'ensemble'
):
    """Run A/B test comparing baseline and personalized ranking."""
    print("=" * 60)
    print(f"Running A/B Test ({num_queries} queries)")
    print("=" * 60)
    
    if model_file is None:
        model_file = config.MODELS_DIR / "personalized_model.json"
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        print("Please train a model first using --mode train")
        return
    
    # Initialize components
    client = UserSimulationClient()
    baseline_ranker = BaselineRanker()
    personalized_ranker = PersonalizedRanker(method=method)
    personalized_ranker.load(model_file)
    
    control_log = config.LOGS_DIR / "ab_control.jsonl"
    treatment_log = config.LOGS_DIR / "ab_treatment.jsonl"
    
    control_logger = InteractionLogger(control_log)
    treatment_logger = InteractionLogger(treatment_log)
    
    # Check if simulation server is running
    if not client.health_check():
        print(f"\n❌ User simulation server is not running at {client.base_url}")
        return
    
    print(f"✓ Connected to simulation server")
    print(f"✓ Loaded personalized model (method: {personalized_ranker.method})")
    print(f"Control (baseline) log: {control_log}")
    print(f"Treatment (personalized) log: {treatment_log}")
    
    # Run A/B test
    control_count = 0
    treatment_count = 0
    
    # Cumulative tracking
    control_clicks = 0
    treatment_clicks = 0
    control_dwell = 0.0
    treatment_dwell = 0.0
    progress_interval = max(1000, num_queries // 10)
    
    for i in tqdm(range(num_queries), desc="Processing queries"):
        try:
            # Get query from simulation
            query = client.get_query()
            
            # Randomly assign to control or treatment (50/50 split)
            is_treatment = random.random() < 0.5
            
            if is_treatment:
                # Treatment: personalized ranking
                ranker = personalized_ranker
                logger = treatment_logger
                ranker_type = "personalized"
                treatment_count += 1
                
                ranked_ids = ranker.get_ranked_ids(
                    query.user_id,
                    query.query_text,
                    top_k=config.RERANK_TOP_K
                )
            else:
                # Control: baseline ranking
                ranker = baseline_ranker
                logger = control_logger
                ranker_type = "baseline"
                control_count += 1
                
                ranked_ids = ranker.get_ranked_ids(
                    query.query_text,
                    top_k=config.RERANK_TOP_K
                )
            
            if not ranked_ids:
                continue
            
            # Submit to simulation and get actions
            response = client.submit_ranklist(query.query_id, query.user_id, ranked_ids)
            
            # Get full article data for logging
            ranked_articles = []
            base_ranker = personalized_ranker.baseline_ranker if is_treatment else baseline_ranker
            for article_id in ranked_ids:
                article = base_ranker.get_article_by_id(article_id)
                if article:
                    article["uuid"] = article_id
                    ranked_articles.append(article)
            
            # Log interaction
            logger.log_interaction(
                user_id=query.user_id,
                query_id=query.query_id,
                query_text=query.query_text,
                ranked_articles=ranked_articles,
                actions=[[a.__dict__ for a in action_list] for action_list in response.actions],
                ranker_type=ranker_type,
                metadata={"ab_variant": "treatment" if is_treatment else "control"}
            )
            
            # Track cumulative stats per variant
            for action_list in response.actions:
                for action in action_list:
                    if action.action_type == "Click":
                        if is_treatment:
                            treatment_clicks += 1
                        else:
                            control_clicks += 1
                    elif action.action_type == "Dwell" and action.dwell_time:
                        if is_treatment:
                            treatment_dwell += action.dwell_time
                        else:
                            control_dwell += action.dwell_time
            
            # Print periodic summary for long runs
            if (i + 1) % progress_interval == 0 and num_queries > 1000:
                tqdm.write(f"\n--- Progress Update ({i+1}/{num_queries} queries) ---")
                tqdm.write(f"  Control:   {control_count} queries, {control_clicks} clicks, {control_dwell:.0f}s dwell")
                tqdm.write(f"  Treatment: {treatment_count} queries, {treatment_clicks} clicks, {treatment_dwell:.0f}s dwell")
            
        except Exception as e:
            print(f"\nError processing query {i+1}: {e}")
            continue
    
    print(f"\nControl queries: {control_count}")
    print(f"Treatment queries: {treatment_count}")
    
    # Print A/B cumulative summary
    print("\n" + "=" * 60)
    print("A/B Test Cumulative Summary")
    print("=" * 60)
    ctrl_ctr = control_clicks / control_count * 100 if control_count > 0 else 0
    treat_ctr = treatment_clicks / treatment_count * 100 if treatment_count > 0 else 0
    print(f"  Control (Baseline):     {control_clicks} clicks ({ctrl_ctr:.2f}% CTR), {control_dwell:.0f}s dwell")
    print(f"  Treatment (Personalized): {treatment_clicks} clicks ({treat_ctr:.2f}% CTR), {treatment_dwell:.0f}s dwell")
    
    # Load interactions
    control_interactions = InteractionLogger.load_logs(control_log)
    treatment_interactions = InteractionLogger.load_logs(treatment_log)
    
    # Run statistical tests
    print("\n" + "=" * 60)
    print("Statistical Analysis")
    print("=" * 60)
    
    ab_test = ABTest()
    results = ab_test.run_test(control_interactions, treatment_interactions)
    
    print(f"\nControl sample size: {results['control']['sample_size']}")
    print(f"Treatment sample size: {results['treatment']['sample_size']}")
    print(f"\nOverall Winner: {results['overall_winner'].upper()}")
    
    print("\n" + "=" * 60)
    print("Metrics Comparison")
    print("=" * 60)
    
    for metric_key in ["ctr", "mrr", "ndcg", "engagement_rate"]:
        control_value = results['control'][metric_key]
        treatment_value = results['treatment'][metric_key]
        improvement = (treatment_value - control_value) / control_value * 100 if control_value > 0 else 0
        
        print(f"\n{metric_key.upper()}:")
        print(f"  Control:   {control_value:.4f}")
        print(f"  Treatment: {treatment_value:.4f}")
        print(f"  Change:    {improvement:+.2f}%")
    
    print("\n" + "=" * 60)
    print("Significance Tests")
    print("=" * 60)
    
    for test in results['significance_tests']:
        print(f"\n{test['metric_name']}:")
        print(f"  Control:    {test['control_mean']:.4f}")
        print(f"  Treatment:  {test['treatment_mean']:.4f}")
        print(f"  Difference: {test['improvement_pct']:+.2f}%")
        print(f"  P-value:    {test['p_value']}")
        print(f"  Significant: {'✓ YES' if test['is_significant'] else '✗ NO'} (α={config.AB_TEST_ALPHA})")
        print(f"  Effect size: {test['effect_size']:.4f}")
    
    # Save results
    results_file = config.LOGS_DIR / "ab_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Full results saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Personalized News Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["setup", "baseline", "train", "personalized", "ab-test"],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to process (default: 100)"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file (for train mode)"
    )
    
    parser.add_argument(
        "--model-file",
        type=Path,
        help="Path to model file"
    )
    
    parser.add_argument(
        "--method",
        choices=["cf", "gbm", "ensemble"],
        default="ensemble",
        help="Ranking method: cf (collaborative filtering), gbm (gradient boosting), or ensemble (default)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        setup_elasticsearch()
    
    elif args.mode == "baseline":
        run_baseline(num_queries=args.num_queries, log_file=args.log_file)
    
    elif args.mode == "train":
        if args.log_file is None:
            args.log_file = config.LOGS_DIR / "baseline_interactions.jsonl"
        train_personalized_model(args.log_file, args.model_file, method=args.method)
    
    elif args.mode == "personalized":
        run_personalized(
            num_queries=args.num_queries,
            model_file=args.model_file,
            log_file=args.log_file,
            method=args.method
        )
    
    elif args.mode == "ab-test":
        run_ab_test(
            num_queries=args.num_queries,
            model_file=args.model_file,
            method=args.method
        )


if __name__ == "__main__":
    main()
