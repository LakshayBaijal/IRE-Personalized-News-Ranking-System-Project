"""Client for interacting with the user simulation API."""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import config


@dataclass
class Query:
    """Represents a user query."""
    user_id: str
    query_id: str
    query_text: str


@dataclass
class Action:
    """Represents a user action on an article."""
    action_type: str
    dwell_time: Optional[float] = None  # in seconds
    
    @classmethod
    def from_dict(cls, action_data: Any) -> 'Action':
        """Parse action from API response."""
        if isinstance(action_data, str):
            return cls(action_type=action_data)
        elif isinstance(action_data, dict) and "Dwell" in action_data:
            dwell_info = action_data["Dwell"]
            dwell_seconds = dwell_info["secs"] + dwell_info["nanos"] / 1e9
            return cls(action_type="Dwell", dwell_time=dwell_seconds)
        else:
            return cls(action_type=str(action_data))


@dataclass
class RankListResponse:
    """Response from ranklist endpoint."""
    query_id: str
    user_id: str
    ranked_article_ids: List[str]
    actions: List[List[Action]]
    
    def get_engagement_score(self, position: int, weights: Dict[str, float] = None) -> float:
        """Calculate engagement score for article at given position."""
        if weights is None:
            weights = config.METRIC_WEIGHTS
        
        if position >= len(self.actions):
            return 0.0
        
        score = 0.0
        article_actions = self.actions[position]
        
        for action in article_actions:
            if action.action_type == "Click":
                score += weights.get("click", 1.0)
            elif action.action_type == "Dwell" and action.dwell_time:
                score += action.dwell_time * weights.get("dwell_time", 0.1)
            elif action.action_type == "Like":
                score += weights.get("like", 5.0)
            elif action.action_type == "Share":
                score += weights.get("share", 10.0)
            elif action.action_type == "Bookmark":
                score += weights.get("bookmark", 7.0)
        
        return score
    
    def has_click_at(self, position: int) -> bool:
        """Check if article at position was clicked."""
        if position >= len(self.actions):
            return False
        return any(a.action_type == "Click" for a in self.actions[position])


class UserSimulationClient:
    """Client for user simulation API."""
    
    def __init__(self, base_url: str = None):
        """Initialize client."""
        self.base_url = base_url or config.USER_SIM_BASE_URL
        
    def get_query(self) -> Query:
        """Get a random user query."""
        response = requests.get(f"{self.base_url}/query", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return Query(
            user_id=data["user_id"],
            query_id=data["query_id"],
            query_text=data["query_text"]
        )
    
    def submit_ranklist(
        self,
        query_id: str,
        user_id: str,
        ranked_article_ids: List[str]
    ) -> RankListResponse:
        """Submit ranked list and get user actions."""
        payload = {
            "query_id": query_id,
            "user_id": user_id,
            "ranked_article_ids": ranked_article_ids
        }
        
        response = requests.post(
            f"{self.base_url}/ranklist",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse actions
        actions = []
        for action_list in data["actions"]:
            parsed_actions = [Action.from_dict(a) for a in action_list]
            actions.append(parsed_actions)
        
        return RankListResponse(
            query_id=query_id,
            user_id=user_id,
            ranked_article_ids=ranked_article_ids,
            actions=actions
        )
    
    def health_check(self) -> bool:
        """Check if the simulation server is running."""
        try:
            response = requests.get(f"{self.base_url}/query", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def main():
    """Test the user simulation client."""
    client = UserSimulationClient()
    
    # Check health
    if not client.health_check():
        print("❌ User simulation server is not running!")
        print(f"Please start it at {client.base_url}")
        return
    
    print("✓ User simulation server is running")
    
    # Test getting a query
    query = client.get_query()
    print(f"\nQuery: {query.query_text}")
    print(f"User ID: {query.user_id}")
    print(f"Query ID: {query.query_id}")
    
    # Test submitting a dummy ranklist
    dummy_article_ids = ["550e8400-e29b-41d4-a716-446655440000"] * 5
    try:
        response = client.submit_ranklist(
            query.query_id,
            query.user_id,
            dummy_article_ids
        )
        print(f"\nReceived {len(response.actions)} action lists")
        for i, actions in enumerate(response.actions):
            if actions:
                print(f"Article {i}: {[a.action_type for a in actions]}")
    except Exception as e:
        print(f"Note: Dummy article IDs may not exist in simulation: {e}")


if __name__ == "__main__":
    main()

