"""Elasticsearch setup and indexing for news articles."""

import json
from typing import List, Dict, Any
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import config


class ElasticsearchSetup:
    """Handle Elasticsearch indexing and setup."""
    
    def __init__(self, host: str = config.ES_HOST, port: int = config.ES_PORT):
        """Initialize Elasticsearch client."""
        self.es = Elasticsearch([f"http://{host}:{port}"])
        self.index_name = config.ES_INDEX_NAME
        
    def create_index(self) -> None:
        """Create index with appropriate mappings."""
        mapping = {
            "mappings": {
                "properties": {
                    "uuid": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "english",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "topics": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "english": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        }
        
        # Delete index if exists
        if self.es.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        # Create new index
        print(f"Creating index: {self.index_name}")
        self.es.indices.create(index=self.index_name, body=mapping)
        
    def load_articles(self, file_path: str = None) -> List[Dict[str, Any]]:
        """Load articles from JSONL file."""
        if file_path is None:
            file_path = config.ARTICLES_FILE
            
        articles = []
        print(f"Loading articles from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():
                    article = json.loads(line)
                    articles.append(article)
        
        print(f"Loaded {len(articles)} articles")
        return articles
    
    def index_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Bulk index articles to Elasticsearch."""
        def generate_actions():
            for article in articles:
                yield {
                    "_index": self.index_name,
                    "_id": article["uuid"],
                    "_source": article
                }
        
        print(f"Indexing {len(articles)} articles...")
        success, failed = helpers.bulk(
            self.es,
            generate_actions(),
            chunk_size=500,
            request_timeout=60
        )
        
        print(f"Indexed {success} articles successfully")
        if failed:
            print(f"Failed to index {len(failed)} articles")
        
        # Refresh index
        self.es.indices.refresh(index=self.index_name)
        
    def verify_index(self) -> None:
        """Verify the index was created successfully."""
        count = self.es.count(index=self.index_name)["count"]
        print(f"Total documents in index: {count}")
        
        # Sample document
        result = self.es.search(
            index=self.index_name,
            body={"query": {"match_all": {}}, "size": 1}
        )
        
        if result["hits"]["hits"]:
            print("\nSample document:")
            print(json.dumps(result["hits"]["hits"][0]["_source"], indent=2))
    
    def setup(self) -> None:
        """Complete setup process."""
        self.create_index()
        articles = self.load_articles()
        self.index_articles(articles)
        self.verify_index()
        print("\n✓ Elasticsearch setup complete!")


def main():
    """Run Elasticsearch setup."""
    try:
        es_setup = ElasticsearchSetup()
        es_setup.setup()
    except Exception as e:
        print(f"Error during setup: {e}")
        raise


if __name__ == "__main__":
    main()

