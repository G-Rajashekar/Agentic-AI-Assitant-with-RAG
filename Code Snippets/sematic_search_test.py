from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

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


embedding1=get_embedding("python is a programming language")
embedding2=get_embedding("python used in AI and ML")
embedding3=get_embedding("bye")

similarity_12=get_cosine_similarity(embedding1,embedding2)
similarity_13=get_cosine_similarity(embedding1,embedding3)

print(f"{similarity_12:.2f}")
print(f"{similarity_13:.2f}")