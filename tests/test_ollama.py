from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


result = embeddings.embed_query(
    "Kinetic energy is the energy of a moving object."
)


print("Embedding created successfully.")
print("Number of values:", len(result))
print("First 10 values:")
print(result[:10])