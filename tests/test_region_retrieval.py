from src.retrieval.textbook_retriever import (
    retrieve_documents,
    documents_to_context,
    retrieve_textbook_context_near_pages
)

from src.retrieval.query_generator import generate_retrieval_queries
from statistics import median

topic = "Types of Chemical Reactions"


print("STEP 1: Initial search...\n")

initial_documents = retrieve_documents(
    topic,
    k_per_query=5
)

initial_context = documents_to_context(initial_documents)


print("STEP 2: Finding topic location...\n")

pages = [
    document.metadata["page"]
    for document in initial_documents
    if document.metadata.get("page") is not None
]

centre_page = round(median(pages))

print("Initial pages:", pages)
print("Estimated centre page:", centre_page)


print("\nSTEP 3: Generating textbook-based queries...\n")

queries = generate_retrieval_queries(
    topic,
    initial_context
)

for query in queries:
    print("-", query)


print("\nSTEP 4: Final region-aware retrieval...\n")

final_context = retrieve_textbook_context_near_pages(
    queries,
    centre_page=centre_page,
    page_radius=10,
    k_per_query=8
)

print(final_context)