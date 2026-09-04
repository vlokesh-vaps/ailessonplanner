from query_generator import generate_retrieval_queries


topic = "Types of Chemical Reactions"

queries = generate_retrieval_queries(topic)

print("Generated retrieval queries:\n")

for query in queries:
    print("-", query)
