from src.services.retrieval import semantic_search, keyword_search, rrf_fusion
from src.services.reranking import simple_rerank
from src.services.generation import generate_answer
from src.services.hallucination_check import check_hallucination, verify_citations

def rag_query(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True,
    check_hallucinations: bool = True
) -> dict:
    """
    Complete RAG pipeline: Retrieve -> Rerank -> Generate -> Verify
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        use_reranking: Whether to rerank results
        check_hallucinations: Whether to check for hallucinations
    
    Returns:
        Complete response with answer, sources, and verification
    """
    
    # Step 1: Hybrid Retrieval
    semantic_results = semantic_search(query, top_k=top_k * 2)
    keyword_results = keyword_search(query, top_k=top_k * 2)
    fused_results = rrf_fusion(semantic_results, keyword_results)
    
    # Step 2: Reranking (optional)
    if use_reranking:
        reranked_results = simple_rerank(query, fused_results, top_k=top_k)
    else:
        reranked_results = fused_results[:top_k]
    
    # Step 3: Answer Generation
    generation_result = generate_answer(query, reranked_results, include_citations=True)
    
    # Step 4: Hallucination Check (optional)
    if check_hallucinations:
        hallucination_result = check_hallucination(
            query,
            generation_result["answer"],
            reranked_results
        )
        citation_verification = verify_citations(
            generation_result["answer"],
            reranked_results
        )
    else:
        hallucination_result = None
        citation_verification = None
    
    # Step 5: Compile complete response
    return {
        "query": query,
        "answer": generation_result["answer"],
        "sources": generation_result["sources"],
        "confidence": generation_result["confidence"],
        "retrieval": {
            "semantic_count": len(semantic_results),
            "keyword_count": len(keyword_results),
            "fused_count": len(fused_results),
            "final_count": len(reranked_results)
        },
        "verification": {
            "hallucination_check": hallucination_result,
            "citations": citation_verification
        } if check_hallucinations else None
    }