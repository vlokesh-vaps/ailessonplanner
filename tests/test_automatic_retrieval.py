from src.retrieval.query_generator import generate_retrieval_queries
from src.retrieval.textbook_retriever import retrieve_textbook_context


topic = "Types of Chemical Reactions"


# ------------------------------------------------------------
# STAGE 1: Initial retrieval using the topic
# ------------------------------------------------------------

print("Retrieving initial textbook context...\n")

initial_context = retrieve_textbook_context(
    [topic],
    k_per_query=3
)

print(initial_context)


# ------------------------------------------------------------
# STAGE 2: Generate better retrieval queries
# ------------------------------------------------------------

print("\nGenerating retrieval queries...\n")

queries = generate_retrieval_queries(
    topic,
    initial_context
)

for query in queries:
    print("-", query)


# ------------------------------------------------------------
# STAGE 3: Retrieve textbook content using improved queries
# ------------------------------------------------------------

print("\nRetrieving final textbook content...\n")

context = retrieve_textbook_context(
    queries,
    k_per_query=3
)


print(context)