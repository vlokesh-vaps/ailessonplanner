from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def generate_retrieval_queries(topic, initial_context):
    prompt = f"""
You are generating retrieval queries for a textbook.

Your task is to identify the main topic and direct subtopics using ONLY
the textbook material provided below.

Rules:
- Use terminology that appears in the textbook material.
- Do not introduce synonyms that are not used in the textbook.
- Do not introduce additional related topics.
- Do not use chapter numbers or section numbers.
- Include the main lesson topic.
- Include the direct subtopics that clearly belong to that topic.
- Generate 4 to 8 short search queries.
- Return one query per line only.
- Do not add explanations.

LESSON TOPIC:
{topic}

TEXTBOOK MATERIAL:
{initial_context}
"""

    response = llm.invoke(prompt)

    queries = []

    for line in response.content.splitlines():
        line = line.strip()

        if line:
            line = line.lstrip("-•0123456789. ").strip()

            if line:
                queries.append(line)

    return queries