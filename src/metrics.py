"""Evaluation metrics for ranking systems."""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import config


@dataclass
class RankingMetrics:
    """Container for ranking metrics."""
    ctr: float  # Click-through rate
    mrr: float  # Mean reciprocal rank
    ndcg: float  # Normalized DCG
    avg_dwell_time: float  # Average dwell time
    engagement_rate: float  # Likes, shares, bookmarks per query
    precision_at_k: float  # Precision at k
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "ctr": self.ctr,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "avg_dwell_time": self.avg_dwell_time,
            "engagement_rate": self.engagement_rate,
            "precision_at_k": self.precision_at_k
        }


class MetricsCalculator:
    """Calculate ranking metrics from logged interactions."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with metric weights."""
        self.weights = weights or config.METRIC_WEIGHTS
    
    def calculate_ctr(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate click-through rate."""
        total_articles = 0
        total_clicks = 0
        
        for interaction in interactions:
            for action_list in interaction["actions"]:
                total_articles += 1
                if any(self._is_click(a) for a in action_list):
                    total_clicks += 1
        
        return total_clicks / max(total_articles, 1)
    
    def calculate_mrr(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate mean reciprocal rank of first click."""
        reciprocal_ranks = []
        
        for interaction in interactions:
            for position, action_list in enumerate(interaction["actions"], 1):
                if any(self._is_click(a) for a in action_list):
                    reciprocal_ranks.append(1.0 / position)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg(
        self,
        interactions: List[Dict[str, Any]],
        k: int = config.RERANK_TOP_K
    ) -> float:
        """Calculate normalized discounted cumulative gain."""
        ndcg_scores = []
        
        for interaction in interactions:
            relevances = []
            for action_list in interaction["actions"][:k]:
                relevance = self._calculate_relevance(action_list)
                relevances.append(relevance)
            
            if not relevances:
                continue
            
            # DCG
            dcg = relevances[0]
            for i, rel in enumerate(relevances[1:], 2):
                dcg += rel / np.log2(i + 1)
            
            # IDCG (ideal DCG with perfect ranking)
            sorted_relevances = sorted(relevances, reverse=True)
            idcg = sorted_relevances[0]
            for i, rel in enumerate(sorted_relevances[1:], 2):
                idcg += rel / np.log2(i + 1)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_avg_dwell_time(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate average dwell time on clicked articles."""
        dwell_times = []
        
        for interaction in interactions:
            for action_list in interaction["actions"]:
                for action in action_list:
                    # Handle new format: {"action_type": "Dwell", "dwell_time": 17.68}
                    if isinstance(action, dict) and action.get("action_type") == "Dwell":
                        if action.get("dwell_time"):
                            dwell_times.append(action["dwell_time"])
                    # Handle old format: {"Dwell": {"secs": 45, "nanos": 0}}
                    elif isinstance(action, dict) and "Dwell" in action:
                        dwell_info = action["Dwell"]
                        dwell_time = dwell_info["secs"] + dwell_info["nanos"] / 1e9
                        dwell_times.append(dwell_time)
        
        return np.mean(dwell_times) if dwell_times else 0.0
    
    def calculate_engagement_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate engagement rate (likes, shares, bookmarks per query)."""
        total_engagements = 0
        
        for interaction in interactions:
            for action_list in interaction["actions"]:
                for action in action_list:
                    action_type = None
                    if isinstance(action, str):
                        action_type = action
                    elif isinstance(action, dict):
                        action_type = action.get("action_type")
                    
                    if action_type in ["Like", "Share", "Bookmark"]:
                        total_engagements += 1
        
        return total_engagements / max(len(interactions), 1)
    
    def calculate_precision_at_k(
        self,
        interactions: List[Dict[str, Any]],
        k: int = config.RERANK_TOP_K
    ) -> float:
        """Calculate precision at k (proportion of clicked articles in top k)."""
        precisions = []
        
        for interaction in interactions:
            clicked = 0
            for action_list in interaction["actions"][:k]:
                if any(self._is_click(a) for a in action_list):
                    clicked += 1
            
            precision = clicked / k
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def calculate_all_metrics(
        self,
        interactions: List[Dict[str, Any]]
    ) -> RankingMetrics:
        """Calculate all metrics."""
        return RankingMetrics(
            ctr=self.calculate_ctr(interactions),
            mrr=self.calculate_mrr(interactions),
            ndcg=self.calculate_ndcg(interactions),
            avg_dwell_time=self.calculate_avg_dwell_time(interactions),
            engagement_rate=self.calculate_engagement_rate(interactions),
            precision_at_k=self.calculate_precision_at_k(interactions)
        )
    
    def _is_click(self, action: Any) -> bool:
        """Check if action is a click."""
        if isinstance(action, str):
            return action == "Click"
        elif isinstance(action, dict):
            return action.get("action_type") == "Click"
        return False
    
    def _calculate_relevance(self, action_list: List[Any]) -> float:
        """Calculate relevance score for an article based on actions."""
        score = 0.0
        
        for action in action_list:
            # Get action type from either format
            action_type = None
            if isinstance(action, str):
                action_type = action
            elif isinstance(action, dict):
                action_type = action.get("action_type")
            
            if action_type == "Click":
                score += self.weights.get("click", 1.0)
            elif action_type == "Like":
                score += self.weights.get("like", 5.0)
            elif action_type == "Share":
                score += self.weights.get("share", 10.0)
            elif action_type == "Bookmark":
                score += self.weights.get("bookmark", 7.0)
            elif action_type == "Dwell":
                # New format: {"action_type": "Dwell", "dwell_time": 17.68}
                if isinstance(action, dict) and action.get("dwell_time"):
                    score += action["dwell_time"] * self.weights.get("dwell_time", 0.1)
            
            # Old format: {"Dwell": {"secs": 45, "nanos": 0}}
            if isinstance(action, dict) and "Dwell" in action:
                dwell_info = action["Dwell"]
                dwell_time = dwell_info["secs"] + dwell_info["nanos"] / 1e9
                score += dwell_time * self.weights.get("dwell_time", 0.1)
        
        return score


def main():
    """Test metrics calculation."""
    # Sample interactions
    interactions = [
        {
            "user_id": "user1",
            "query_id": "query1",
            "actions": [
                ["Click", {"Dwell": {"secs": 45, "nanos": 0}}, "Like"],
                [],
                ["Click", {"Dwell": {"secs": 20, "nanos": 0}}]
            ]
        },
        {
            "user_id": "user2",
            "query_id": "query2",
            "actions": [
                [],
                ["Click", {"Dwell": {"secs": 30, "nanos": 0}}],
                []
            ]
        }
    ]
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(interactions)
    
    print("Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

