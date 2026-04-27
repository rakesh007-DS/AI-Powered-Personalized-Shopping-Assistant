"""
Evaluation metrics for the shopping assistant
"""
from typing import List, Dict
import json

def relevance_at_k(recommendations: List[Dict], relevant_items: List[str], k: int = 3) -> float:
    top_k = recommendations[:k]
    hits = sum(1 for item in top_k if item.get('product_id') in relevant_items)
    return hits / k

def mean_average_precision(all_recommendations: List[List[Dict]], all_relevant: List[List[str]]) -> float:
    ap_scores = []
    for recs, relevant in zip(all_recommendations, all_relevant):
        hits = 0
        precision_sum = 0
        for i, rec in enumerate(recs):
            if rec.get('product_id') in relevant:
                hits += 1
                precision_sum += hits / (i + 1)
        ap = precision_sum / len(relevant) if relevant else 0
        ap_scores.append(ap)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0

def run_evaluation(rag_system, test_queries: List[Dict]) -> Dict:
    results = {
        "relevance_at_3": [],
        "response_times": [],
        "map_score": 0
    }

    all_recs, all_relevant = [], []

    for query_data in test_queries:
        import time
        start = time.time()
        recs = rag_system.retrieve_products(query_data['query'])
        elapsed = time.time() - start

        results["response_times"].append(elapsed)
        r_at_3 = relevance_at_k(recs, query_data['relevant_ids'])
        results["relevance_at_3"].append(r_at_3)
        all_recs.append(recs)
        all_relevant.append(query_data['relevant_ids'])

    results["map_score"] = mean_average_precision(all_recs, all_relevant)
    results["avg_relevance_at_3"] = sum(results["relevance_at_3"]) / len(results["relevance_at_3"])
    results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])

    print(f"Relevance@3: {results['avg_relevance_at_3']:.2%}")
    print(f"MAP Score: {results['map_score']:.2%}")
    print(f"Avg Response Time: {results['avg_response_time']:.3f}s")
    return results

if __name__ == "__main__":
    print("Run evaluation with your RAG system instance.")
