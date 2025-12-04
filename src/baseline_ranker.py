"""Baseline ranking using Elasticsearch BM25."""

from typing import List, Dict, Any
from elasticsearch import Elasticsearch
import config


class BaselineRanker:
    """Baseline ranker using Elasticsearch BM25."""
    
    def __init__(self, host: str = config.ES_HOST, port: int = config.ES_PORT):
        """Initialize Elasticsearch client."""
        self.es = Elasticsearch([f"http://{host}:{port}"])
        self.index_name = config.ES_INDEX_NAME
        
    def search(self, query: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        """
        Search for articles using BM25.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of article documents with scores
        """
        search_body = {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "or"
                    }
                }
            },
            "size": top_k,
            "_source": ["uuid", "text", "topics"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "uuid": hit["_source"]["uuid"],
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "topics": hit["_source"]["topics"]
            })
        
        return results
    
    def get_ranked_ids(self, query: str, top_k: int = config.RERANK_TOP_K) -> List[str]:
        """
        Get ranked article IDs for a query.
        
        Args:
            query: Search query text
            top_k: Number of article IDs to return
            
        Returns:
            List of article UUIDs in ranked order
        """
        results = self.search(query, top_k=top_k)
        return [r["uuid"] for r in results]
    
    def get_article_by_id(self, article_id: str) -> Dict[str, Any]:
        """Get article document by ID."""
        try:
            response = self.es.get(index=self.index_name, id=article_id)
            return response["_source"]
        except Exception:
            return None


def main():
    """Test the baseline ranker."""
    ranker = BaselineRanker()
    
    # Test query
    test_query = "technology innovation"
    print(f"Testing query: '{test_query}'")
    
    results = ranker.search(test_query, top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Topics: {', '.join(result['topics'])}")
        print(f"   Text: {result['text'][:150]}...")


if __name__ == "__main__":
    main()

