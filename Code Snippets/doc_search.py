import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBED_MODEL")
KB_FOLDER=os.getenv("KB_FOLDER")

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


def add_embeddings(documents):
    for doc in documents:
        vector=get_embedding(doc["text"])
        doc["vector"]=vector

    return documents


def get_top_k_docs(query,documents,top_k):
    query_vector=get_embedding(query)

    results=[]

    for doc in documents:
        score=get_cosine_similarity(query_vector,doc["vector"])

        results.append(
            {
                "score":float(score),
                "source":doc["source"],
                "text":doc["text"],
                "size":len(doc["text"])
            }
        )

    results.sort(key=lambda rec:rec["score"],reverse=True)

    return results[:top_k]

def test(user_query,documents,top_k=3):
    print(f"user query: {user_query}")

    top_k_results=get_top_k_docs(user_query,documents,top_k)

    for rec in top_k_results:
        print(f"score: {rec["score"]:.2f} source: {rec["source"]} size: {rec["size"]}")


documents=load_documents()
documents=add_embeddings(documents)

print(f"we have embedded {len(documents)} documents")

user_query="How many days I will get for maternity leave"

test(user_query,documents,3)