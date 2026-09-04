from pathlib import Path

import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "inputs" / "NotesExampleIn.pdf"
CHROMA_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_db"
COLLECTION_NAME = "textbook"

CHROMA_PATH.mkdir(parents=True, exist_ok=True)


print("STEP 1: Loading PDF...")

pdf = pymupdf.open(str(PDF_PATH))

pages = []

for page_number, page in enumerate(pdf):
    text = page.get_text()

    if text.strip():
        document = Document(
            page_content=text,
            metadata={
                "source": str(PDF_PATH),
                "page": page_number
            }
        )

        pages.append(document)

pdf.close()

print(f"Loaded {len(pages)} pages containing text.")


print("\nSTEP 2: Splitting PDF into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks.")


print("\nSTEP 3: Loading Ollama embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


print("\nSTEP 4: Creating Chroma vector database...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_name=COLLECTION_NAME
)


print("\nVector database created successfully.")
print(f"Stored in: {CHROMA_PATH}")
