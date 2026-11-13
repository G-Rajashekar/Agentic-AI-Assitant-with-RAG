import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBED_MODEL")

def get_embedding(text):
    response=client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )

    return response.data[0].embedding

chunks={
    "c0":"Leave policy: Maternity leave is 26 weeks of paid leave for the first two children.",
    "c1":"Medical policy: Employees get full health insurance coverage.",
    "c2":"Travel expense: Travel expenses are reimbursed upon submission of receipts.",
    "c3":"Tech docs: Production deployments are allowed on Tuesday and Thursday between 11 AM and 4 PM."
}

ids=list(chunks.keys())
documents=list(chunks.values())
embeddings=[]

for s in chunks:
    embedding=get_embedding(s)
    embeddings.append(embedding)

chroma=chromadb.PersistentClient(path="./chromadbStorage")

zorvex_collection=chroma.get_or_create_collection(
    name="zorvex_technology",
    configuration={
        "hnsw":{
            "space":"cosine"
        }
    }
)

zorvex_collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

query="when is the production deployement allowed?"

query_embedding=get_embedding(query)
results=zorvex_collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(results["documents"][0])