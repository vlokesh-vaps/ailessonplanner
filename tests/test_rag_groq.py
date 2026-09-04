from dotenv import load_dotenv
from langchain_groq import ChatGroq

from textbook_retriever import retrieve_textbook_context


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


topic = "types of chemical reactions combination decomposition displacement"


context = retrieve_textbook_context(
    topic,
    k=5
)


prompt = f"""
You are an educational assistant.

Using only the textbook material below, explain the topic clearly.

TOPIC:
{topic}

TEXTBOOK MATERIAL:
{context}
"""


response = llm.invoke(prompt)

print(response.content)