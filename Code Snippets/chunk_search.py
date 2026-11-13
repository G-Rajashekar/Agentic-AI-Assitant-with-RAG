import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import json

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBED_MODEL")
KB_FOLDER=os.getenv("KB_FOLDER")
VECTOR_STORE_FILE=os.getenv("VECTOR_STORE_FILE")

def get_embedding(text):
    response=client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )

    return response.data[0].embedding

def get_cosine_similarity(A,B):

    dot_product=np.dot(A,B)

    length_A=np.linalg.norm(A)
    length_B=np.linalg.norm(B)

    return dot_product/(length_A*length_B)

def load_documents(folder=KB_FOLDER):
    documents=[]

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path=os.path.join(folder,filename)

        with open(path,"r",encoding="utf-8") as f:
            documents.append(
                {
                    "source":filename,
                    "text":f.read()
                }
            )

    return documents 

def add_embeddings(chunks):
    for chunk in chunks:
        vector=get_embedding(chunk["text"])
        chunk["vector"]=vector

    return chunks

def get_top_k_chunks(query,chunks,top_k):

    query_vector=get_embedding(query)

    results=[]

    for chunk in chunks:
        score=get_cosine_similarity(query_vector,chunk["vector"])
        results.append(
            {
                "score":float(score),
                "source":chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "text":chunk["text"],
                "size":chunk["size"]
            }
        )

    results.sort(key=lambda rec:rec["score"],reverse=True)

    return results[:top_k]

def get_chunks_from_document(document):
    chunks=[]
    chunk_id=0
    for chunk in document["text"].split("\n\n"):
        chunk=chunk.strip()
        if chunk:
            chunks.append(
                {
                    "source":document["source"],
                    "chunk_id":chunk_id,
                    "text":chunk,
                    "size":len(chunk)
                }
            )
            chunk_id+=1

    return chunks

def create_chunks():
    documents=load_documents()
    chunks=[]
    for document in documents:
        chunks.extend(get_chunks_from_document(document))

    return chunks

def test(user_query,chunks,top_k=3):
    print(f"user query: {user_query}")

    top_k_results=get_top_k_chunks(user_query,chunks,top_k)

    for rec in top_k_results:
        print(f"score: {rec["score"]:.2f} source: {rec["source"]} size: {rec["size"]}")
        print(f"  {rec["text"]}")

def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE,"w",encoding="utf-8") as f:
        json.dump(chunks,f,indent=4)

def load_vector_store():
    with open(VECTOR_STORE_FILE,"r",encoding="utf-8") as f:
        return json.load(f)


if os.path.exists(VECTOR_STORE_FILE):
    chunks=load_vector_store()
    print(f"Loaded {len(chunks)} chunks from vector store file {VECTOR_STORE_FILE}")
else:
    chunks=create_chunks()
    chunks=add_embeddings(chunks)
    save_vector_store(chunks)
    print(f"Created and Embedded {len(chunks)} chunks and saved to vector store file {VECTOR_STORE_FILE}")

user_query="what are the benefits of maternity leave ?"

test(user_query,chunks,3)