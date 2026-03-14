import math 
import voyageai
from dotenv import load_dotenv
import os 

load_dotenv()

client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

A = "Python is a great programming language"
B = "I enjoy coding in Python"  
C = "The stock market crashed today"

response = client.embed(
    [A,B,C],
    model="voyage-3-lite"
)

embedding_A = response.embeddings[0]
embedding_B= response.embeddings[1]
embedding_C= response.embeddings[2]

def cosine_similariy(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

print(cosine_similariy(embedding_A,embedding_B))
print(cosine_similariy(embedding_A,embedding_C))
print(cosine_similariy(embedding_B,embedding_C))