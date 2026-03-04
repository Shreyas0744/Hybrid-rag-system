from src.services.ingestion import ingest_pdf

ingest_pdf(
    pdf_path="sample.pdf",
    document_id="test_document"
)

print("PDF ingestion completed")
