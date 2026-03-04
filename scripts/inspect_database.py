# scripts/inspect_database.py
from src.database.session import get_session
from sqlalchemy import text

def inspect_database():
    session = get_session()
    
    # Count total chunks
    total = session.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
    print(f"Total chunks: {total}")
    
    # Check for duplicates
    duplicates = session.execute(text("""
        SELECT content, COUNT(*) as count
        FROM document_chunks
        GROUP BY content
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10;
    """)).fetchall()
    
    if duplicates:
        print(f"\n⚠️ Found {len(duplicates)} duplicate content entries:")
        for i, dup in enumerate(duplicates, 1):
            print(f"{i}. Count: {dup.count} | Content: {dup.content[:100]}...")
    else:
        print("\n✅ No duplicate content found")
    
    # Show first 5 chunks
    print("\n--- First 5 Chunks ---")
    chunks = session.execute(text("""
        SELECT id, content
        FROM document_chunks
        LIMIT 5;
    """)).fetchall()
    
    for chunk in chunks:
        print(f"\nID {chunk.id}:")
        print(chunk.content[:200])
    
    session.close()

if __name__ == "__main__":
    inspect_database()