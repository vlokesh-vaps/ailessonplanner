from src.retrieval.query_generator import generate_retrieval_queries


topic = "Types of Chemical Reactions"

initial_context = """
Types of chemical reactions include combination reactions,
decomposition reactions, displacement reactions,
double displacement reactions, oxidation and reduction.
Some reactions may also be exothermic or endothermic.
"""


queries = generate_retrieval_queries(
    topic,
    initial_context
)


print("Generated retrieval queries:\n")

for query in queries:
    print("-", query)