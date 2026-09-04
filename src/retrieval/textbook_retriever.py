from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def retrieve_documents(queries, k_per_query=3):
    if isinstance(queries, str):
        queries = [queries]

    all_documents = []
    seen_chunks = set()

    for query in queries:
        documents = vector_db.similarity_search(
            query,
            k=k_per_query
        )

        for document in documents:
            chunk_key = (
                document.metadata.get("page"),
                document.page_content
            )

            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                all_documents.append(document)

    return all_documents


def documents_to_context(documents):
    context_parts = []

    for document in documents:
        page = document.metadata.get("page", "unknown")

        context_parts.append(
            f"""
SOURCE PAGE: {page}

{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


def retrieve_textbook_context(queries, k_per_query=3):
    documents = retrieve_documents(
        queries,
        k_per_query=k_per_query
    )

    return documents_to_context(documents)


def retrieve_textbook_context_near_pages(
    queries,
    centre_page,
    page_radius=10,
    k_per_query=8
):
    if isinstance(queries, str):
        queries = [queries]

    all_documents = []
    seen_chunks = set()

    minimum_page = centre_page - page_radius
    maximum_page = centre_page + page_radius

    for query in queries:

        # Retrieve more candidates than we ultimately need.
        candidates = vector_db.similarity_search(
            query,
            k=k_per_query
        )

        for document in candidates:
            page = document.metadata.get("page")

            if page is None:
                continue

            # Keep only material near the identified textbook region.
            if minimum_page <= page <= maximum_page:

                chunk_key = (
                    page,
                    document.page_content
                )

                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    all_documents.append(document)

    return documents_to_context(all_documents)