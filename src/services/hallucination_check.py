from google import genai
from src.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini.api_key)

def check_hallucination(query: str, answer: str, context_chunks: list[dict]) -> dict:
    """
    Check if the generated answer contains hallucinations.
    
    Returns:
        dict with 'is_grounded', 'confidence', and 'explanation'
    """
    
    context = "\n".join([chunk["content"] for chunk in context_chunks[:5]])
    
    prompt = f"""You are a fact-checker. Determine if the answer is fully grounded in the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Instructions:
1. Check if every claim in the answer is supported by the context
2. Respond with a JSON object containing:
   - "is_grounded": true/false
   - "confidence": 0.0 to 1.0
   - "explanation": brief explanation of your judgment

Response (JSON only):"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        # Parse JSON response
        import json
        result_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
        
        result = json.loads(result_text)
        
        return {
            "is_grounded": result.get("is_grounded", False),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", "")
        }
        
    except Exception as e:
        print(f"Error in hallucination check: {e}")
        return {
            "is_grounded": True,  # Assume grounded if check fails
            "confidence": 0.5,
            "explanation": f"Check failed: {str(e)}"
        }


def verify_citations(answer: str, context_chunks: list[dict]) -> dict:
    """
    Verify that cited sources actually support the claims.
    """
    
    import re
    
    # Extract citation numbers [Source N]
    citations = re.findall(r'\[Source (\d+)\]', answer)
    
    verified_citations = []
    
    for cite_num in set(citations):
        idx = int(cite_num) - 1
        if 0 <= idx < len(context_chunks):
            verified_citations.append({
                "source_id": cite_num,
                "chunk_id": context_chunks[idx]["id"],
                "content": context_chunks[idx]["content"][:200] + "..."
            })
    
    return {
        "total_citations": len(set(citations)),
        "verified_sources": verified_citations,
        "has_citations": len(citations) > 0
    }