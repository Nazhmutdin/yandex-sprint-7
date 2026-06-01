from os import listdir
from pathlib import Path
from time import time_ns

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("script started")

model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

client = chromadb.PersistentClient("./Task 3/chroma")

collection = client.get_or_create_collection("diablo_collection")

print("infra created")

PROCESSED = Path(__file__).parent / "Task 2" / "knowledge_base" / "processed"

print("start processing")

start = time_ns()

chunk_count = 0

for file in listdir(PROCESSED):
    file_path = PROCESSED / file

    print(f"file {file} processed")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = splitter.split_text(f.read())

        for e, chunk in enumerate(chunks, start=1):
            embeddings = model.encode(chunk).tolist()

            collection.add(
                ids=[f"{file}-{e}"],
                embeddings=embeddings,
                documents=[chunk],
                metadatas=[{"file_path": str(file_path)}],
            )

            chunk_count += 1


print(f"execution time takes: {(time_ns() - start) / 1_000_000_000}")

print(f"chunk amount: {chunk_count}")
