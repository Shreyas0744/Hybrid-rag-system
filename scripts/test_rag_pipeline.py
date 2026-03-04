from src.services.rag_pipeline import rag_query
import json

def test_rag():
    queries = [
        "What is the internship duration?",
        "What are the terms and conditions?",
        "Who should I contact?",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        result = rag_query(
            query=query,
            top_k=5,
            use_reranking=True,
            check_hallucinations=True
        )
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']:.3f}")
        
        if result['verification']:
            hallucination = result['verification']['hallucination_check']
            print(f"\nGrounded: {hallucination['is_grounded']}")
            print(f"Explanation: {hallucination['explanation']}")
        
        print(f"\nSources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['content'][:100]}...")

if __name__ == "__main__":
    test_rag()