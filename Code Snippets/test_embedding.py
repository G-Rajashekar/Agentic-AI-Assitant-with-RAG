from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response=client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=["python is good language","I am testing embedding","working on rag"]
)

for i,emb in enumerate(response.data):
    print(f"Embedding {i+1} length: {len(emb.embedding)}")
    print(emb.embedding[:5])