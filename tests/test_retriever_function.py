from src.retrieval.textbook_retriever import retrieve_textbook_context


queries = [
    "1.2 Types of Chemical Reactions",
    "1.2.1 Combination Reaction",
    "1.2.2 Decomposition Reaction",
    "1.2.3 Displacement Reaction",
    "1.2.4 Double Displacement Reaction"
]


context = retrieve_textbook_context(
    queries,
    k_per_query=4
)

print(context)