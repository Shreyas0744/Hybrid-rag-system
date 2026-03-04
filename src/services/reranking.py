from google import genai
from src.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini.api_key)

def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using cross-encoder or LLM-based reranking.
    
    For now, we'll use a simple LLM-based relevance scoring.
    For production, consider using a dedicated reranking model.
    """
    
    if len(chunks) <= top_k:
        return chunks
    
    reranked = []
    
    for chunk in chunks:
        relevance_score = _compute_relevance(query, chunk["content"])
        reranked.append({
            **chunk,
            "rerank_score": relevance_score,
            "original_score": chunk["score"]
        })
    
    # Sort by rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked[:top_k]


def _compute_relevance(query: str, content: str) -> float:
    """
    Compute relevance score using LLM judgment.
    Returns a score between 0 and 1.
    """
    
    prompt = f"""Rate how relevant this content is to answering the question.
Respond with ONLY a number between 0 and 1, where:
- 0 = completely irrelevant
- 0.5 = somewhat relevant
- 1 = highly relevant and directly answers the question

Question: {query}

Content: {content[:500]}

Relevance score:"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        # Extract score from response
        score_text = response.text.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        
    except Exception as e:
        print(f"Error in relevance scoring: {e}")
        return 0.5  # Default to medium relevance


def simple_rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Simple reranking based on keyword overlap (no API calls).
    Faster but less accurate than LLM-based reranking.
    """
    
    query_terms = set(query.lower().split())
    
    for chunk in chunks:
        content_terms = set(chunk["content"].lower().split())
        overlap = len(query_terms & content_terms)
        chunk["keyword_overlap"] = overlap
        # Combine original score with keyword overlap
        chunk["rerank_score"] = chunk["score"] * 0.7 + (overlap / len(query_terms)) * 0.3
    
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]