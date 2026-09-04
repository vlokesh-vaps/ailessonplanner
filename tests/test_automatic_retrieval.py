from query_generator import generate_retrieval_queries
from textbook_retriever import retrieve_textbook_context


topic = "Types of Chemical Reactions"


print("Generating retrieval queries...\n")

queries = generate_retrieval_queries(topic)

for query in queries:
    print("-", query)


print("\nRetrieving textbook content...\n")

context = retrieve_textbook_context(
    queries,
    k_per_query=3
)


print(context)