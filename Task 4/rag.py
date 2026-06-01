import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from sentence_transformers import SentenceTransformer

load_dotenv("./.env")

MODEL = "o4-mini"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient("./Task 3/chroma")

collection = client.get_or_create_collection("diablo_collection")


def get_system_prompt(context: list[str]):
    return f"""
You are an expert on the Diablo universe who thinks first, then answers. Always descrybe your steps.

**Instructions:**
1. Analyze the context from the knowledge base: <CONTEXT_START>\n\n\n{"\n".join(context)}\n\n\n<CONTEXT_END>
2. Answer the question as specifically as possible
3. If the context does not contain enough data to answer, return only what is available, otherwise: "Sorry, there is no information in the knowledge base to answer your question"
4. For complex questions, break down the answer into points

**Answer style:**
- Do not use emoji and maintain a technical or near-technical tone
- Highlight important parts of the answer

**Example answer:**
Q: Who is Aelorin?
Thought: First I will examine the context from the knowledge base
Thought: The context states that "Aelorin is a rogue angel who, together with Morraia, created Vaerath"
A: Aelorin is a fallen angel who betrayed the The Celestial Vaults, joined forces with Morraia, and created Vaerath
"""


client = OpenAI(
    api_key=os.getenv("API_KEY"), base_url="https://api.proxyapi.ru/openai/v1"
)


while True:
    user_query = input("Q: ")

    query_embeddings = model.encode(user_query).tolist()

    result = collection.query(query_embeddings)

    documents = result["documents"]

    if documents is None:
        print("No data found")
        continue

    documents = documents[0]

    system_prompt = get_system_prompt(list(reversed(documents))[:7])

    messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_query),
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_completion_tokens=40000
    )

    print(response.choices[0].message.content, "\n\n\n")
