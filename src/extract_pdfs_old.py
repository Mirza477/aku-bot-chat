import os
import uuid
import pdfplumber
from tabulate import tabulate
from config.settings import PDF_DIR
from .cosmos_db import upsert_policy_section
from .embeddings import generate_embedding

def extract_text_and_tables(pdf_path: str):
    """Extract text and tables from a single PDF file."""
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            sections.append({
                "text": text.strip(),
                "tables": tables
            })
    return sections

def process_pdfs():
    """
    1) For each PDF in PDF_DIR, extract text & tables
    2) Generate embeddings
    3) Upsert them into Cosmos DB
    """
    if not os.path.isdir(PDF_DIR):
        print(f"PDF directory does not exist: {PDF_DIR}")
        return
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        doc_name = os.path.basename(pdf_path)
        sections = extract_text_and_tables(pdf_path)

        # For each page or "section," combine text + table in one chunk
        for idx, sec in enumerate(sections, start=1):
            content = sec["text"]
            if sec["tables"]:
                for table in sec["tables"]:
                    content += "\n\nTable:\n"
                    content += tabulate(table, tablefmt="pipe")

            # Generate embedding
            embedding = generate_embedding(content)

            # Upsert item into Cosmos
            item = {
                "id": f"{doc_name}-{uuid.uuid4()}",
                "document_name": doc_name,
                "section": f"Section {idx}",
                "content": content,
                "vector": embedding
            }
            upsert_policy_section(item)
            print(f"Upserted: {doc_name} - Section {idx}")

if __name__ == "__main__":
    process_pdfs()
