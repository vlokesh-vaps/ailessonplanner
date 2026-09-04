from src.retrieval.textbook_retriever import retrieve_textbook_context
from src.retrieval.query_generator import generate_retrieval_queries


topic = "Types of Chemical Reactions"


print("STEP 1: Initial textbook search...\n")

initial_context = retrieve_textbook_context(
    topic,
    k_per_query=5
)


print("STEP 2: Generating textbook-based retrieval queries...\n")

queries = generate_retrieval_queries(
    topic,
    initial_context
)

for query in queries:
    print("-", query)


print("\nSTEP 3: Retrieving final textbook context...\n")

final_context = retrieve_textbook_context(
    queries,
    k_per_query=3
)

print(final_context)
