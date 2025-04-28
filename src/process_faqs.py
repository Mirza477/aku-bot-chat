import os
import uuid
import pdfplumber
import re
from config.settings import FAQ_DIR
from .cosmos_db import upsert_policy_section
from .embeddings import generate_embedding

import re

def extract_metadata_from_first_page(pdf_path: str):
    """
    Extract metadata such as title, version, and last_updated from the first page of a PDF.
    """
    metadata = {"title": None, "version": None, "last_updated": None}
    with pdfplumber.open(pdf_path) as pdf:
        if pdf.pages:
            first_page_text = pdf.pages[0].extract_text() or ""
            print("DEBUG: First page text:", first_page_text)  # Debug print
            # Use regex to find metadata entries
            title_match = re.search(r"Document Title:\s*(.*)", first_page_text, re.IGNORECASE)
            version_match = re.search(r"Version:\s*(\S+)", first_page_text, re.IGNORECASE)
            updated_match = re.search(r"Last Updated:\s*(\S+)", first_page_text, re.IGNORECASE)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
            if version_match:
                metadata["version"] = version_match.group(1).strip()
            if updated_match:
                metadata["last_updated"] = updated_match.group(1).strip()

            print("DEBUG: Extracted metadata:", metadata)  # Debug print
    return metadata


def extract_faqs_from_pdf(pdf_path: str):
    """
    Extracts FAQ entries and metadata from a PDF.
    """
    metadata = extract_metadata_from_first_page(pdf_path)
    
    faqs = []
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += "\n" + text

    # Normalize whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    # Pattern for FAQs; adjust based on your document format.
    faq_pattern = re.compile(r'(Q:\s*.*?)(?=Q:|$)', re.IGNORECASE)
    matches = faq_pattern.findall(full_text)
    
    for match in matches:
        parts = re.split(r'A:\s*', match, flags=re.IGNORECASE)
        if len(parts) >= 2:
            question = parts[0].strip()
            answer = "A: " + parts[1].strip()
            # Remove "Q:" marker
            question = re.sub(r'^Q:\s*', '', question, flags=re.IGNORECASE)
            faqs.append({"question": question, "answer": answer})
    return metadata, faqs

def process_faqs():
    if not os.path.isdir(FAQ_DIR):
        print(f"FAQ directory does not exist: {FAQ_DIR}")
        return

    faq_files = [f for f in os.listdir(FAQ_DIR) if f.lower().endswith(".pdf")]
    if not faq_files:
        print(f"No FAQ PDF files found in: {FAQ_DIR}")
        return

    for faq_file in faq_files:
        file_path = os.path.join(FAQ_DIR, faq_file)
        document_name = os.path.basename(file_path)
        metadata, faqs = extract_faqs_from_pdf(file_path)
        
        if not faqs:
            print(f"No FAQ entries found in: {document_name}")
            continue
        
        for idx, faq in enumerate(faqs, start=1):
            faq_title = f"FAQ: {faq['question']}"
            faq_content = f"Title: {metadata.get('title', document_name)}\n" \
                          f"Version: {metadata.get('version', 'N/A')}\n" \
                          f"Last Updated: {metadata.get('last_updated', 'N/A')}\n\n" \
                          f"Q: {faq['question']}\nA: {faq['answer']}"
            try:
                embedding = generate_embedding(faq_content)
            except Exception as e:
                print(f"Error generating embedding for FAQ from {document_name}: {e}")
                continue

            item = {
                "id": f"{document_name}-{uuid.uuid4()}",
                "document_name": document_name,
                "section": faq_title,
                "content": faq_content,
                "vector": embedding,
                "version": metadata.get("version", "N/A"),
                "last_updated": metadata.get("last_updated", "N/A"),
                "category": "FAQ"
            }
            try:
                upsert_policy_section(item)
                print(f"Upserted FAQ Entry: {faq_title} from {document_name}")
            except Exception as e:
                print(f"Error upserting FAQ Entry {faq_title}: {e}")

if __name__ == "__main__":
    process_faqs()
