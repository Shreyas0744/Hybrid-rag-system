# scripts/clear_database.py
from src.database.session import get_session
from sqlalchemy import text

def clear_database():
    session = get_session()
    try:
        session.execute(text("DELETE FROM document_chunks;"))
        session.commit()
        print("✅ Database cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    clear_database()