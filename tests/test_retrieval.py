from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = str(PROJECT_ROOT / "data" / "vector_db" / "chroma_db")
COLLECTION_NAME = "textbook"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)


query = "pH scale acids bases neutralisation"


print(f"Searching textbook for: {query}\n")


results = vector_db.similarity_search(
    query,
    k=5
)


for number, document in enumerate(results, start=1):

    print("=" * 80)
    print(f"RESULT {number}")
    print("Page:", document.metadata.get("page"))
    print()
    print(document.page_content)
    print()