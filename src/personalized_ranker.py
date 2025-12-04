"""Personalized ranking using classical ML and recommendation system methods."""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import config
from src.baseline_ranker import BaselineRanker
from src.logger import InteractionLogger


class CollaborativeFilteringModel:
    """User-based and item-based collaborative filtering."""
    
    def __init__(self, k_neighbors: int = 10):
        """Initialize CF model."""
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_similarities = {}
        self.item_similarities = {}
        self.user_id_map = {}
        self.article_id_map = {}
        self.user_means = {}
        
    def fit(self, interactions: List[Dict[str, Any]]) -> None:
        """Train collaborative filtering model."""
        # Build user-item interaction matrix
        user_article_scores = defaultdict(lambda: defaultdict(float))
        
        for interaction in interactions:
            user_id = interaction['user_id']
            ranked_articles = interaction['ranked_articles']
            actions = interaction['actions']
            
            for article, action_list in zip(ranked_articles, actions):
                article_id = article.get('uuid')
                if not article_id:
                    continue
                
                # Calculate engagement score
                score = self._calculate_engagement(action_list)
                if score > 0:
                    user_article_scores[user_id][article_id] += score
        
        # Create mappings
        users = sorted(user_article_scores.keys())
        all_articles = set()
        for articles in user_article_scores.values():
            all_articles.update(articles.keys())
        articles = sorted(all_articles)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.article_id_map = {aid: idx for idx, aid in enumerate(articles)}
        
        # Build matrix
        n_users = len(users)
        n_items = len(articles)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for user_id, articles_dict in user_article_scores.items():
            user_idx = self.user_id_map[user_id]
            for article_id, score in articles_dict.items():
                if article_id in self.article_id_map:
                    item_idx = self.article_id_map[article_id]
                    self.user_item_matrix[user_idx, item_idx] = score
        
        # Calculate user means
        for user_id in users:
            user_idx = self.user_id_map[user_id]
            ratings = self.user_item_matrix[user_idx]
            ratings = ratings[ratings > 0]
            self.user_means[user_id] = np.mean(ratings) if len(ratings) > 0 else 0.0
        
        # Compute user-user similarity (cosine)
        self._compute_user_similarities()
        
        print(f"CF Model trained: {n_users} users, {n_items} articles")
    
    def _compute_user_similarities(self) -> None:
        """Compute user-user cosine similarities."""
        n_users = self.user_item_matrix.shape[0]
        
        # Normalize by user mean
        matrix_centered = self.user_item_matrix.copy()
        for i in range(n_users):
            ratings = matrix_centered[i]
            if np.sum(ratings > 0) > 0:
                mean = np.mean(ratings[ratings > 0])
                matrix_centered[i][ratings > 0] -= mean
        
        # Cosine similarity
        norms = np.linalg.norm(matrix_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix_centered / norms
        
        # Store top-k similarities for each user
        similarities = matrix_normalized @ matrix_normalized.T
        
        for i in range(n_users):
            # Get top k similar users (excluding self)
            similar_indices = np.argsort(similarities[i])[::-1][1:self.k_neighbors+1]
            similar_scores = similarities[i][similar_indices]
            self.user_similarities[i] = list(zip(similar_indices, similar_scores))
    
    def predict(self, user_id: str, article_id: str) -> float:
        """Predict user's interest in an article."""
        if user_id not in self.user_id_map or article_id not in self.article_id_map:
            return 0.0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.article_id_map[article_id]
        
        # If user has already interacted with this article
        if self.user_item_matrix[user_idx, item_idx] > 0:
            return self.user_item_matrix[user_idx, item_idx]
        
        # Collaborative filtering prediction
        if user_idx not in self.user_similarities:
            return self.user_means.get(user_id, 0.0)
        
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_idx, similarity in self.user_similarities[user_idx]:
            neighbor_rating = self.user_item_matrix[neighbor_idx, item_idx]
            if neighbor_rating > 0:
                numerator += similarity * neighbor_rating
                denominator += abs(similarity)
        
        if denominator > 0:
            return numerator / denominator
        
        return self.user_means.get(user_id, 0.0)
    
    def _calculate_engagement(self, action_list: List[Any]) -> float:
        """Calculate engagement score from actions."""
        score = 0.0
        weights = config.METRIC_WEIGHTS
        
        for action in action_list:
            if isinstance(action, str):
                if action == "Click":
                    score += weights.get("click", 1.0)
                elif action == "Like":
                    score += weights.get("like", 5.0)
                elif action == "Share":
                    score += weights.get("share", 10.0)
                elif action == "Bookmark":
                    score += weights.get("bookmark", 7.0)
            elif isinstance(action, dict) and "Dwell" in action:
                dwell_info = action["Dwell"]
                dwell_time = dwell_info["secs"] + dwell_info["nanos"] / 1e9
                score += dwell_time * weights.get("dwell_time", 0.1)
        
        return score


class FeatureBasedRanker:
    """Learning-to-rank using gradient boosting (LightGBM)."""
    
    def __init__(self):
        """Initialize feature-based ranker."""
        self.model = None
        self.scaler = StandardScaler()
        self.user_topic_affinity = defaultdict(lambda: defaultdict(float))
        self.global_topic_popularity = defaultdict(float)
        self.article_engagement = defaultdict(float)
        self.feature_names = []
        
    def fit(self, interactions: List[Dict[str, Any]]) -> None:
        """Train learning-to-rank model."""
        # Extract features and labels
        features, labels, groups = self._extract_features_labels(interactions)
        
        if len(features) == 0:
            print("No training data available")
            return
        
        print(f"Training LightGBM with {len(features)} samples, {len(groups)} queries")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train LightGBM ranker
        train_data = lgb.Dataset(X, label=y, group=groups)
        
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        print(f"Model trained successfully")
        
        # Print feature importance
        importance = self.model.feature_importance()
        for fname, imp in zip(self.feature_names, importance):
            if imp > 0:
                print(f"  {fname}: {imp:.1f}")
    
    def _extract_features_labels(
        self,
        interactions: List[Dict[str, Any]]
    ) -> tuple:
        """Extract features and relevance labels from interactions."""
        features = []
        labels = []
        groups = []  # Number of samples per query (for LambdaRank)
        
        # First pass: compute statistics
        self._compute_statistics(interactions)
        
        # Second pass: extract features
        for interaction in interactions:
            user_id = interaction['user_id']
            ranked_articles = interaction['ranked_articles']
            actions = interaction['actions']
            
            query_features = []
            query_labels = []
            
            for article, action_list in zip(ranked_articles, actions):
                # Extract features
                feat = self._extract_article_features(user_id, article)
                
                # Calculate relevance label (engagement score)
                label = self._calculate_engagement(action_list)
                
                query_features.append(feat)
                query_labels.append(label)
            
            if len(query_features) > 0:
                features.extend(query_features)
                labels.extend(query_labels)
                groups.append(len(query_features))
        
        return features, labels, groups
    
    def _compute_statistics(self, interactions: List[Dict[str, Any]]) -> None:
        """Compute user-topic affinity and global statistics."""
        topic_counts = defaultdict(int)
        topic_engagement = defaultdict(float)
        article_engagement_total = defaultdict(float)
        article_counts = defaultdict(int)
        
        for interaction in interactions:
            user_id = interaction['user_id']
            ranked_articles = interaction['ranked_articles']
            actions = interaction['actions']
            
            for article, action_list in zip(ranked_articles, actions):
                article_id = article.get('uuid')
                topics = article.get('topics', [])
                engagement = self._calculate_engagement(action_list)
                
                # Update statistics
                if article_id:
                    article_engagement_total[article_id] += engagement
                    article_counts[article_id] += 1
                
                for topic in topics:
                    topic_counts[topic] += 1
                    topic_engagement[topic] += engagement
                    
                    if engagement > 0:
                        self.user_topic_affinity[user_id][topic] += engagement
        
        # Normalize
        for user_id in self.user_topic_affinity:
            total = sum(self.user_topic_affinity[user_id].values())
            if total > 0:
                for topic in self.user_topic_affinity[user_id]:
                    self.user_topic_affinity[user_id][topic] /= total
        
        # Global topic popularity
        total_engagement = sum(topic_engagement.values())
        if total_engagement > 0:
            for topic in topic_engagement:
                self.global_topic_popularity[topic] = topic_engagement[topic] / total_engagement
        
        # Article engagement
        for article_id in article_engagement_total:
            count = article_counts[article_id]
            self.article_engagement[article_id] = article_engagement_total[article_id] / count
    
    def _extract_article_features(
        self,
        user_id: str,
        article: Dict[str, Any]
    ) -> List[float]:
        """Extract features for a user-article pair."""
        features = []
        
        article_id = article.get('uuid', '')
        topics = article.get('topics', [])
        bm25_score = article.get('score', 0.0)
        
        # Feature 1: BM25 score (baseline relevance)
        features.append(bm25_score)
        
        # Feature 2-3: User-topic affinity (max and mean)
        topic_affinities = [
            self.user_topic_affinity[user_id].get(topic, 0.0)
            for topic in topics
        ]
        features.append(max(topic_affinities) if topic_affinities else 0.0)
        features.append(np.mean(topic_affinities) if topic_affinities else 0.0)
        
        # Feature 4-5: Global topic popularity (max and mean)
        topic_popularities = [
            self.global_topic_popularity.get(topic, 0.0)
            for topic in topics
        ]
        features.append(max(topic_popularities) if topic_popularities else 0.0)
        features.append(np.mean(topic_popularities) if topic_popularities else 0.0)
        
        # Feature 6: Article historical engagement
        features.append(self.article_engagement.get(article_id, 0.0))
        
        # Feature 7: Number of topics
        features.append(len(topics))
        
        # Feature 8: User has history with any topic
        user_has_topic = any(
            topic in self.user_topic_affinity[user_id]
            for topic in topics
        )
        features.append(1.0 if user_has_topic else 0.0)
        
        # Store feature names (only once)
        if len(self.feature_names) == 0:
            self.feature_names = [
                'bm25_score',
                'user_topic_affinity_max',
                'user_topic_affinity_mean',
                'global_topic_popularity_max',
                'global_topic_popularity_mean',
                'article_engagement',
                'num_topics',
                'user_has_topic'
            ]
        
        return features
    
    def predict(self, user_id: str, article: Dict[str, Any]) -> float:
        """Predict relevance score for user-article pair."""
        if self.model is None:
            return 0.0
        
        features = self._extract_article_features(user_id, article)
        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        
        return self.model.predict(features)[0]
    
    def _calculate_engagement(self, action_list: List[Any]) -> float:
        """Calculate engagement score from actions."""
        score = 0.0
        weights = config.METRIC_WEIGHTS
        
        for action in action_list:
            if isinstance(action, str):
                if action == "Click":
                    score += weights.get("click", 1.0)
                elif action == "Like":
                    score += weights.get("like", 5.0)
                elif action == "Share":
                    score += weights.get("share", 10.0)
                elif action == "Bookmark":
                    score += weights.get("bookmark", 7.0)
            elif isinstance(action, dict) and "Dwell" in action:
                dwell_info = action["Dwell"]
                dwell_time = dwell_info["secs"] + dwell_info["nanos"] / 1e9
                score += dwell_time * weights.get("dwell_time", 0.1)
        
        return score


class PersonalizedRanker:
    """Personalized ranker combining multiple classical ML approaches."""
    
    def __init__(
        self,
        baseline_ranker: Optional[BaselineRanker] = None,
        method: str = 'ensemble'  # 'cf', 'gbm', or 'ensemble'
    ):
        """
        Initialize personalized ranker.
        
        Args:
            baseline_ranker: Baseline Elasticsearch ranker
            method: Ranking method ('cf', 'gbm', or 'ensemble')
        """
        self.baseline_ranker = baseline_ranker or BaselineRanker()
        self.method = method
        self.cf_model = CollaborativeFilteringModel()
        self.gbm_model = FeatureBasedRanker()
        self.is_trained = False
        
    def train(self, log_file: Path) -> None:
        """Train the ranking models from logs."""
        interactions = InteractionLogger.load_logs(log_file)
        
        print(f"\nTraining personalized ranker with method: {self.method}")
        print(f"Learning from {len(interactions)} interactions...")
        
        if self.method in ['cf', 'ensemble']:
            print("\nTraining Collaborative Filtering model...")
            self.cf_model.fit(interactions)
        
        if self.method in ['gbm', 'ensemble']:
            print("\nTraining Gradient Boosting model...")
            self.gbm_model.fit(interactions)
        
        self.is_trained = True
        print("\n✓ Training complete!")
    
    def rerank(
        self,
        user_id: str,
        query: str,
        baseline_results: List[Dict[str, Any]] = None,
        top_k: int = config.RERANK_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Rerank articles using trained models.
        
        Args:
            user_id: User identifier
            query: Search query
            baseline_results: Optional pre-computed baseline results
            top_k: Number of results to return
            
        Returns:
            Reranked list of articles
        """
        # Get baseline results if not provided
        if baseline_results is None:
            baseline_results = self.baseline_ranker.search(query, top_k=config.TOP_K)
        
        if not self.is_trained:
            # Return baseline if not trained
            return baseline_results[:top_k]
        
        # Calculate personalized scores
        for article in baseline_results:
            base_score = article['score']
            
            # Get predictions from different models
            cf_score = 0.0
            gbm_score = 0.0
            
            if self.method in ['cf', 'ensemble']:
                article_id = article.get('uuid', '')
                cf_score = self.cf_model.predict(user_id, article_id)
            
            if self.method in ['gbm', 'ensemble']:
                gbm_score = self.gbm_model.predict(user_id, article)
            
            # Combine scores based on method
            if self.method == 'cf':
                personalized_score = cf_score
            elif self.method == 'gbm':
                personalized_score = gbm_score
            else:  # ensemble
                # Weighted combination
                personalized_score = 0.5 * cf_score + 0.5 * gbm_score
            
            # Blend with baseline (60% baseline, 40% personalization)
            combined_score = 0.6 * base_score + 0.4 * personalized_score
            
            article['personalized_score'] = combined_score
            article['cf_score'] = cf_score
            article['gbm_score'] = gbm_score
        
        # Sort by personalized score
        reranked = sorted(
            baseline_results,
            key=lambda x: x['personalized_score'],
            reverse=True
        )
        
        return reranked[:top_k]
    
    def get_ranked_ids(
        self,
        user_id: str,
        query: str,
        top_k: int = config.RERANK_TOP_K
    ) -> List[str]:
        """Get ranked article IDs with personalization."""
        results = self.rerank(user_id, query, top_k=top_k)
        return [r["uuid"] for r in results]
    
    def save(self, model_path: Path) -> None:
        """Save the trained models."""
        model_data = {
            'method': self.method,
            'is_trained': self.is_trained,
            'cf_user_id_map': self.cf_model.user_id_map,
            'cf_article_id_map': self.cf_model.article_id_map,
            'cf_user_item_matrix': self.cf_model.user_item_matrix.tolist() if self.cf_model.user_item_matrix is not None else None,
            'cf_user_means': self.cf_model.user_means,
            'cf_user_similarities': {str(k): v for k, v in self.cf_model.user_similarities.items()},
            'gbm_user_topic_affinity': dict(self.gbm_model.user_topic_affinity),
            'gbm_global_topic_popularity': dict(self.gbm_model.global_topic_popularity),
            'gbm_article_engagement': dict(self.gbm_model.article_engagement),
            'gbm_feature_names': self.gbm_model.feature_names
        }
        
        # Save JSON data
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Save LightGBM model separately
        if self.gbm_model.model is not None:
            lgb_path = model_path.parent / f"{model_path.stem}_lgb.txt"
            self.gbm_model.model.save_model(str(lgb_path))
        
        # Save scaler
        if self.gbm_model.scaler is not None:
            import joblib
            scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
            joblib.dump(self.gbm_model.scaler, scaler_path)
        
        print(f"Saved models to {model_path}")
    
    def load(self, model_path: Path) -> None:
        """Load trained models."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.method = model_data['method']
        self.is_trained = model_data['is_trained']
        
        # Load CF model
        self.cf_model.user_id_map = model_data['cf_user_id_map']
        self.cf_model.article_id_map = model_data['cf_article_id_map']
        if model_data['cf_user_item_matrix']:
            self.cf_model.user_item_matrix = np.array(model_data['cf_user_item_matrix'])
        self.cf_model.user_means = model_data['cf_user_means']
        self.cf_model.user_similarities = {int(k): v for k, v in model_data['cf_user_similarities'].items()}
        
        # Load GBM model data
        self.gbm_model.user_topic_affinity = defaultdict(lambda: defaultdict(float), model_data['gbm_user_topic_affinity'])
        self.gbm_model.global_topic_popularity = defaultdict(float, model_data['gbm_global_topic_popularity'])
        self.gbm_model.article_engagement = defaultdict(float, model_data['gbm_article_engagement'])
        self.gbm_model.feature_names = model_data['gbm_feature_names']
        
        # Load LightGBM model
        lgb_path = model_path.parent / f"{model_path.stem}_lgb.txt"
        if lgb_path.exists():
            self.gbm_model.model = lgb.Booster(model_file=str(lgb_path))
        
        # Load scaler
        import joblib
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
        if scaler_path.exists():
            self.gbm_model.scaler = joblib.load(scaler_path)
        
        print(f"Loaded models from {model_path}")


def main():
    """Test the personalized ranker."""
    # Create a dummy log file
    logger = InteractionLogger(config.LOGS_DIR / "test_interactions.jsonl")
    logger.log_interaction(
        user_id="user123",
        query_id="query456",
        query_text="technology news",
        ranked_articles=[
            {"uuid": "art1", "topics": ["science and technology"], "score": 0.9},
            {"uuid": "art2", "topics": ["business"], "score": 0.8}
        ],
        actions=[
            ["Click", {"Dwell": {"secs": 60, "nanos": 0}}, "Like"],
            []
        ]
    )
    
    # Train model
    ranker = PersonalizedRanker(method='ensemble')
    ranker.train(logger.log_file)
    
    # Save model
    model_path = config.MODELS_DIR / "test_model.json"
    ranker.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
