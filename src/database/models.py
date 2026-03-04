from sqlalchemy import (
    Column,
    Integer,
    Text,
    String,
    JSON,
    Index,
    Computed
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)

    # Core content
    content = Column(Text, nullable=False)

    # Vector embedding
    embedding = Column(Vector(768), nullable=False)

    # Metadata
    document_id = Column(String, nullable=False, index=True)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    chunk_metadata = Column(JSON, nullable=True)

    # Full-text search column (FIXED)
    tsv = Column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', content)",
            persisted=True
        )
    )

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": 100},
        ),
        Index(
            "ix_document_chunks_fts",
            "tsv",
            postgresql_using="gin"
        ),
    )
