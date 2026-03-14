import chromadb
from dotenv import load_dotenv
import voyageai
import os 

load_dotenv()

#crete the db(runs locally)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name = "financial_news")

#documents 
documents = [
    "Tesla reported record revenue of $25 billion in Q4 2024",
    "Apple stock rose 3% after strong iPhone sales report",
    "Federal Reserve raised interest rates by 0.25% in January",
    "BYD electric vehicle sales surpassed Tesla globally in 2023",
    "Microsoft Azure cloud revenue grew 28% year over year",
    "Oil prices dropped to $70 per barrel amid recession fears",
    "Amazon reported $143 billion in quarterly revenue",
    "Bitcoin reached $95000 as institutional adoption increased",
    "Nvidia GPU demand surged due to AI training requirements",
    "Goldman Sachs predicts S&P 500 will reach 6500 by end of 2025"
]

#embed documents at once 
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
response = voyage_client.embed(documents,model="voyage-3-lite")
embeddings = response.embeddings

#store in vector db
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))] #unique id for each
)

print(f"Stored {collection.count()} documents in vector database")

#search by meaning 
query = "How did Tesla perform financially?"
query1 = "What is happening with AI and chips?"
query2 = "Tell me about cryptocurrency"

all_queries = [query, query1, query2]
query_embeddings = voyage_client.embed(all_queries, model="voyage-3-lite").embeddings
results = collection.query(
    query_embeddings=[query_embeddings[0]],
    n_results=3     #return top 3 most similar documents
)

results1 = collection.query(
    query_embeddings=[query_embeddings[1]],
    n_results=3     #return top 3 most similar documents
)

results2 = collection.query(
    query_embeddings=[query_embeddings[2]],
    n_results=3     #return top 3 most similar documents
)

print(f"\n Query: '{query}'")
print(f"\nTop 3 results:")
for i, doc in enumerate(results['documents'][0]):
    print(f" {i+1}. {doc}")

print(f"\n Query: '{query1}'")
print(f"\nTop 3 results:")
for i, doc in enumerate(results1['documents'][0]):
    print(f" {i+1}. {doc}")

print(f"\n Query: '{query2}'")
print(f"\nTop 3 results:")
for i, doc in enumerate(results2['documents'][0]):
    print(f" {i+1}. {doc}")