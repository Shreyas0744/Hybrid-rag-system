from src.services.retrieval import semantic_search, keyword_search, rrf_fusion

query = "test"

semantic = semantic_search(query)
keyword = keyword_search(query)

print("Semantic results:", len(semantic))
print("Keyword results:", len(keyword))

results = rrf_fusion(semantic, keyword)

print("Final fused results:", len(results))

for r in results:
    print(r["score"], r["content"][:200])
