from google import genai
from src.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini.api_key)

def generate_answer(query: str, context_chunks: list[dict], include_citations: bool = True) -> dict:
    """
    Generate an answer using retrieved context chunks.
    
    Args:
        query: User's question
        context_chunks: List of retrieved chunks with content and scores
        include_citations: Whether to include source citations
    
    Returns:
        dict with 'answer', 'sources', and 'confidence'
    """
    
    # Build context from chunks
    context = "\n\n".join([
        f"[Source {i+1}] {chunk['content']}"
        for i, chunk in enumerate(context_chunks[:5])  # Top 5 chunks
    ])
    
    # Create prompt with instructions
    if include_citations:
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY information from the context above
2. Cite sources using [Source N] notation
3. If the context doesn't contain enough information, say "I don't have enough information to answer this question fully"
4. Be concise and accurate

Answer:"""
    else:
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY information from the context above
2. If the context doesn't contain enough information, say so clearly
3. Be concise and accurate

Answer:"""
    
    try:
        # Generate response
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        answer = response.text
        
        # Extract sources used (simple heuristic based on citation presence)
        sources_used = []
        for i, chunk in enumerate(context_chunks[:5]):
            if f"[Source {i+1}]" in answer:
                sources_used.append({
                    "id": chunk["id"],
                    "content": chunk["content"][:200] + "...",
                    "score": chunk["score"]
                })
        
        return {
            "answer": answer,
            "sources": sources_used if sources_used else context_chunks[:3],
            "confidence": _calculate_confidence(context_chunks),
            "query": query
        }
        
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "query": query
        }


def _calculate_confidence(chunks: list[dict]) -> float:
    """
    Calculate confidence score based on retrieval scores.
    Simple heuristic: average of top 3 scores.
    """
    if not chunks:
        return 0.0
    
    top_scores = [chunk["score"] for chunk in chunks[:3]]
    return sum(top_scores) / len(top_scores)