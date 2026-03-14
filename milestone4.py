import chromadb
import voyageai
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="financial_news")

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

#store documents 
embeddings = voyage_client.embed(documents, model="voyage-3-lite").embeddings
collection.add(documents=documents, embeddings=embeddings, ids=[f"doc_{i}" for i in range(len(documents))])

#RAG function
def rag_query(question: str) -> str:

    #retrieve
    query_embedding = voyage_client.embed([question], model="voyage-3-lite").embeddings[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_docs = results['documents'][0]

    #augment
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    prompt = f"""You are financial analyst. Answer the question using ONLY the provided context. If the answer isn't in the context, say "I don't have that information."

CONTEXT: {context}

QUESTION: {question}

Cite which facts you used in your answer."""
    

    #generate - LLM answers using context
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text, retrieved_docs

#testing 
question = "How is Tesla performing and who are their competitors?"
answer, sources = rag_query(question)

print(f"Question: {question}")
print(f"\nAnswer:\n{answer}")
print(f"\nSources used:")
for doc in sources:
    print(f"  • {doc}")