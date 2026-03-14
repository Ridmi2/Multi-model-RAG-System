import voyageai
from dotenv import load_dotenv
import os 

load_dotenv()

client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

#getting the embedding(vector) for a piece of text
response = client.embed(
    ["I love programming in Python"],
    model="voyage-3-lite"
)

embedding = response.embeddings[0]

print(f"Number of dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"Type: {type(embedding)}")
