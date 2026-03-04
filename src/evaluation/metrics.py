from typing import List, Dict
import json
from datetime import datetime

def evaluate_retrieval(test_cases: List[Dict]) -> Dict:
    """
    Evaluate retrieval quality using test cases.
    
    Each test case should have:
    - query: str
    - relevant_doc_ids: List[int]
    - retrieved_results: List[dict]
    """
    
    metrics = {
        "precision": [],
        "recall": [],
        "mrr": [],  # Mean Reciprocal Rank
        "ndcg": []  # Normalized Discounted Cumulative Gain
    }
    
    for case in test_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_doc_ids"])
        retrieved = case["retrieved_results"]
        retrieved_ids = [r["id"] for r in retrieved]
        
        # Precision@K
        relevant_retrieved = len([rid for rid in retrieved_ids if rid in relevant_ids])
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        metrics["precision"].append(precision)
        
        # Recall
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
        metrics["recall"].append(recall)
        
        # MRR
        mrr = 0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                mrr = 1 / (i + 1)
                break
        metrics["mrr"].append(mrr)
    
    # Aggregate
    return {
        "avg_precision": sum(metrics["precision"]) / len(metrics["precision"]) if metrics["precision"] else 0,
        "avg_recall": sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0,
        "avg_mrr": sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0,
        "num_queries": len(test_cases)
    }


def evaluate_generation(test_cases: List[Dict]) -> Dict:
    """
    Evaluate answer generation quality.
    
    Each test case should have:
    - query: str
    - generated_answer: str
    - ground_truth: str (optional)
    - context_chunks: List[dict]
    """
    
    from src.services.hallucination_check import check_hallucination
    
    results = []
    
    for case in test_cases:
        hallucination_check = check_hallucination(
            case["query"],
            case["generated_answer"],
            case["context_chunks"]
        )
        
        results.append({
            "query": case["query"],
            "is_grounded": hallucination_check["is_grounded"],
            "confidence": hallucination_check["confidence"]
        })
    
    grounded_count = sum(1 for r in results if r["is_grounded"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0
    
    return {
        "grounding_rate": grounded_count / len(results) if results else 0,
        "avg_confidence": avg_confidence,
        "num_answers": len(results),
        "results": results
    }


def save_evaluation_report(metrics: Dict, output_file: str = "evaluation_report.json"):
    """Save evaluation metrics to file."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Evaluation report saved to {output_file}")