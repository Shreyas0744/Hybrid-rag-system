from sqlalchemy import text
from collections import defaultdict
from src.database.session import get_session
from src.services.embeddings import embed_texts

# -----------------------------
# 1. Semantic Search (pgvector)
# -----------------------------
def semantic_search(query: str, top_k: int = 5):
    session = get_session()
    query_embedding = embed_texts([query])[0]
    
    embedding_str = f"[{','.join(map(str, query_embedding))}]"
    
    # Use DISTINCT ON to get unique content
    sql = text("""
        WITH ranked_chunks AS (
            SELECT DISTINCT ON (content)
                id,
                content,
                1 - (embedding <=> CAST(:query_embedding AS vector)) AS score,
                embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM document_chunks
            ORDER BY content, distance
        )
        SELECT id, content, score
        FROM ranked_chunks
        ORDER BY score DESC
        LIMIT :top_k;
    """)
    
    rows = session.execute(
        sql,
        {
            "query_embedding": embedding_str,
            "top_k": top_k
        }
    ).fetchall()
    session.close()
    
    return [
        {"id": r.id, "content": r.content, "score": float(r.score)}
        for r in rows
    ]

# -----------------------------
# 2. Keyword Search (BM25)
# -----------------------------
def keyword_search(query: str, top_k: int = 5):
    session = get_session()
    
    sql = text("""
        WITH ranked_chunks AS (
            SELECT DISTINCT ON (content)
                id,
                content,
                ts_rank_cd(tsv, plainto_tsquery('english', :query)) AS score
            FROM document_chunks
            WHERE tsv @@ plainto_tsquery('english', :query)
            ORDER BY content, score DESC
        )
        SELECT id, content, score
        FROM ranked_chunks
        ORDER BY score DESC
        LIMIT :top_k;
    """)
    
    rows = session.execute(
        sql,
        {"query": query, "top_k": top_k}
    ).fetchall()
    session.close()
    
    return [
        {"id": r.id, "content": r.content, "score": float(r.score)}
        for r in rows
    ]

# -----------------------------
# 3. Reciprocal Rank Fusion
# -----------------------------
def rrf_fusion(semantic_results, keyword_results, k: int = 60):
    scores = defaultdict(float)
    contents = {}
    
    for rank, item in enumerate(semantic_results):
        scores[item["id"]] += 1 / (k + rank)
        contents[item["id"]] = item["content"]
    
    for rank, item in enumerate(keyword_results):
        scores[item["id"]] += 1 / (k + rank)
        contents[item["id"]] = item["content"]
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {"id": doc_id, "content": contents[doc_id], "score": score}
        for doc_id, score in ranked
    ]