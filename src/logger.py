"""Logging system for user interactions."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import config


class InteractionLogger:
    """Log user interactions with ranked articles."""
    
    def __init__(self, log_file: Path = None):
        """Initialize logger."""
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = config.LOGS_DIR / f"interactions_{timestamp}.jsonl"
        
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_interaction(
        self,
        user_id: str,
        query_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Dict[str, Any]]],
        ranker_type: str = "baseline",
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Log a single interaction.
        
        Args:
            user_id: User identifier
            query_id: Query identifier
            query_text: The search query
            ranked_articles: List of ranked articles with their features
            actions: List of action lists for each article
            ranker_type: Type of ranker used (baseline, personalized, etc.)
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query_id": query_id,
            "query_text": query_text,
            "ranker_type": ranker_type,
            "ranked_articles": ranked_articles,
            "actions": actions,
            "metadata": metadata or {}
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    @staticmethod
    def load_logs(log_file: Path) -> List[Dict[str, Any]]:
        """Load interactions from a log file."""
        interactions = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))
        
        return interactions
    
    @staticmethod
    def merge_logs(log_files: List[Path], output_file: Path) -> None:
        """Merge multiple log files into one."""
        all_interactions = []
        
        for log_file in log_files:
            all_interactions.extend(InteractionLogger.load_logs(log_file))
        
        # Sort by timestamp
        all_interactions.sort(key=lambda x: x["timestamp"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for interaction in all_interactions:
                f.write(json.dumps(interaction) + '\n')
        
        print(f"Merged {len(all_interactions)} interactions into {output_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the log file."""
        if not self.log_file.exists():
            return {"total_interactions": 0}
        
        interactions = self.load_logs(self.log_file)
        
        total_clicks = 0
        total_likes = 0
        total_shares = 0
        total_bookmarks = 0
        total_dwell_time = 0.0
        
        users = set()
        queries = set()
        
        for interaction in interactions:
            users.add(interaction["user_id"])
            queries.add(interaction["query_id"])
            
            for action_list in interaction["actions"]:
                for action in action_list:
                    # Handle both formats: string or dict with action_type
                    action_type = None
                    if isinstance(action, str):
                        action_type = action
                    elif isinstance(action, dict):
                        action_type = action.get("action_type")
                    
                    if action_type == "Click":
                        total_clicks += 1
                    elif action_type == "Like":
                        total_likes += 1
                    elif action_type == "Share":
                        total_shares += 1
                    elif action_type == "Bookmark":
                        total_bookmarks += 1
                    elif action_type == "Dwell":
                        # Handle new format: {"action_type": "Dwell", "dwell_time": 17.68}
                        if isinstance(action, dict) and action.get("dwell_time"):
                            total_dwell_time += action["dwell_time"]
                    
                    # Also handle old format: {"Dwell": {"secs": 45, "nanos": 0}}
                    if isinstance(action, dict) and "Dwell" in action:
                        dwell_info = action["Dwell"]
                        total_dwell_time += dwell_info["secs"] + dwell_info["nanos"] / 1e9
        
        return {
            "total_interactions": len(interactions),
            "unique_users": len(users),
            "unique_queries": len(queries),
            "total_clicks": total_clicks,
            "total_likes": total_likes,
            "total_shares": total_shares,
            "total_bookmarks": total_bookmarks,
            "total_dwell_time": total_dwell_time,
            "avg_dwell_time": total_dwell_time / max(total_clicks, 1)
        }


def main():
    """Test the logger."""
    logger = InteractionLogger()
    
    # Log a sample interaction
    logger.log_interaction(
        user_id="test_user_123",
        query_id="test_query_456",
        query_text="machine learning",
        ranked_articles=[
            {"uuid": "article_1", "topics": ["science and technology"], "score": 0.95},
            {"uuid": "article_2", "topics": ["business"], "score": 0.85}
        ],
        actions=[
            ["Click", {"Dwell": {"secs": 45, "nanos": 0}}, "Like"],
            []
        ],
        ranker_type="baseline"
    )
    
    print(f"Logged to: {logger.log_file}")
    
    # Get statistics
    stats = logger.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

