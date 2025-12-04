"""Debug script to check article UUID matching between ES and simulation."""

import json
import requests
from elasticsearch import Elasticsearch
from src.user_simulation import UserSimulationClient
from src.baseline_ranker import BaselineRanker
import config


def main():
    """Check if article UUIDs match between ES and simulation."""
    print("=" * 60)
    print("Debugging Article UUID Matching")
    print("=" * 60)
    
    # Initialize components
    client = UserSimulationClient()
    ranker = BaselineRanker()
    
    # Check simulation server
    if not client.health_check():
        print("❌ Simulation server not running")
        return
    
    print("✓ Simulation server running\n")
    
    # Get a sample query from simulation
    print("Getting sample query from simulation...")
    query = client.get_query()
    print(f"Query: '{query.query_text}'")
    print(f"User ID: {query.user_id}")
    print(f"Query ID: {query.query_id}\n")
    
    # Search Elasticsearch
    print("Searching Elasticsearch...")
    results = ranker.search(query.query_text, top_k=10)
    print(f"Found {len(results)} results from Elasticsearch\n")
    
    if results:
        print("Sample Elasticsearch results:")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. UUID: {result['uuid'][:40]}...")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Topics: {', '.join(result['topics'])}")
            print(f"   Text: {result['text'][:80]}...")
            print()
        
        # Submit to simulation
        print("Submitting to simulation...")
        ranked_ids = [r['uuid'] for r in results]
        
        try:
            response = client.submit_ranklist(query.query_id, query.user_id, ranked_ids)
            
            print(f"✓ Submission successful\n")
            print("User actions received:")
            
            has_actions = False
            for i, (article_id, actions) in enumerate(zip(ranked_ids, response.actions), 1):
                if actions:
                    has_actions = True
                    print(f"{i}. Article {article_id[:40]}...")
                    print(f"   Actions: {[a.action_type for a in actions]}")
            
            if not has_actions:
                print("⚠️  NO ACTIONS on any article!")
                print("\nPossible causes:")
                print("1. Article UUIDs don't match simulation's dataset")
                print("2. Articles are not relevant to the query")
                print("3. User simulation parameters")
                
                # Check if articles exist in original file
                print("\nChecking articles.jsonl...")
                with open(config.ARTICLES_FILE, 'r') as f:
                    article_uuids = set()
                    for line in f:
                        if line.strip():
                            article = json.loads(line)
                            article_uuids.add(article['uuid'])
                
                print(f"Total articles in articles.jsonl: {len(article_uuids)}")
                
                # Check if ES UUIDs are in the file
                es_uuids = set(r['uuid'] for r in results)
                matching = es_uuids & article_uuids
                print(f"Matching UUIDs: {len(matching)}/{len(es_uuids)}")
                
                if len(matching) == len(es_uuids):
                    print("✓ All ES article UUIDs exist in articles.jsonl")
                    print("\n⚠️  Issue: User simulation is not engaging with these articles")
                    print("This could be normal behavior - try with more diverse queries")
                else:
                    print("❌ UUID mismatch detected!")
            else:
                print("\n✓ User actions detected! System is working correctly.")
                
        except Exception as e:
            print(f"❌ Error submitting to simulation: {e}")
    
    else:
        print("❌ No results from Elasticsearch")
        
        # Check ES index
        es = Elasticsearch([f"http://{config.ES_HOST}:{config.ES_PORT}"])
        count = es.count(index=config.ES_INDEX_NAME)["count"]
        print(f"Articles in ES index: {count}")


if __name__ == "__main__":
    main()

