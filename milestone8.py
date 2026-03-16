from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import chromadb
import voyageai
import os 
import json

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

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

embeddings = voyage_client.embed(documents, model="voyage-3-lite").embeddings
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range (len(documents))]
)

# State 
class FinancialResearchState(TypedDict):
    question: str
    subtasks: list
    retrieved_documents: list
    draft_report: str
    critique: str
    iteration_count: int

# palnner node
def planner_node(state: FinancialResearchState) -> dict:
    print("Planner Agent Running")
    response = llm.invoke(f"""You are a financial research planner.
Break this question into 3-4 specific research subtasks.
Return ONLY a raw JSON array of simple strings.
No markdown, no code fences, no objects.
Start with [ and end with ]

Question: {state['question']}""")     
    try:
        cleaned = response.content.strip()
        subtasks = json.loads(cleaned)
        if subtasks and isinstance(subtasks[0], dict):
            subtasks = [t.get('subtask', str(t)) for t in subtasks]
    except json.JSONDecodeError:
        subtasks = [state['question']]

    print(f" {len(subtasks)} subtasks created")
    return {"subtasks": subtasks, "iteration_count": 0}
  

# Researcher node 

SIMILARITY_THRESHOLD = 0.4

def researcher_node(state: FinancialResearchState) -> dict:
    print("\n Researcher Agent Running...")    

    all_documents = []
    subtasks = state["subtasks"]

    all_embeddings = voyage_client.embed(subtasks, model="voyage-3-lite").embeddings

    for i, subtask in enumerate(subtasks):
        print(f"Searching : '{subtask[:50]}'")
        query_embedding = all_embeddings[i]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "distances"]
        )

        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = (2 - distance) / 2
            if similarity >= SIMILARITY_THRESHOLD:
                all_documents.append({
                    "subtask": subtask,
                    "document": doc,
                    "similarity": round(similarity, 3)
                })
                print(f"   Found (similarity={round(similarity,3)})")

    print(f"{len(all_documents)} documents retrieved")
    return {"retrieved_documents": all_documents}

# ── SYNTHESIZER NODE ───────────────────────────────────
def synthesizer_node(state: FinancialResearchState) -> dict:
    print("\n Synthesizer Agent running...")

    retrieved = state["retrieved_documents"]
    question = state["question"]

    # Handle case where little/no data was found
    if len(retrieved) == 0:
        return {"draft_report": "INSUFFICIENT DATA: No relevant documents found. Research needs to be expanded."}

    # Format documents for the prompt
    context_parts = []
    for item in retrieved:
        context_parts.append(
            f"[Subtask: {item['subtask'][:40]}...]\n"
            f"[Relevance: {item['similarity']}]\n"
            f"[Fact]: {item['document']}"
        )
    context = "\n\n".join(context_parts)

    # Count coverage
    covered_subtasks = set(item['subtask'] for item in retrieved)
    missing_count = len(state['subtasks']) - len(covered_subtasks)

    prompt = f"""You are a financial analyst writing a research report.

QUESTION: {question}

VERIFIED FACTS (use ONLY these):
{context}

INSTRUCTIONS:
1. Write a professional 3-4 paragraph report answering the question
2. Only use facts from the VERIFIED FACTS section above
3. If some aspects of the question lack data, explicitly state: 
   "Note: Insufficient data available for [topic]"
4. End with a "Sources:" section listing facts you used
5. Be honest about gaps — never fabricate data

Write the report now:"""

    response = llm.invoke(prompt)

    print("Draft report generated")
    return {"draft_report": response.content}

# ── BUILD GRAPH ────────────────────────────────────────
graph_builder = StateGraph(FinancialResearchState)
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("synthesizer", synthesizer_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "researcher")
graph_builder.add_edge("researcher", "synthesizer")
graph_builder.add_edge("synthesizer", END)

graph = graph_builder.compile()

# ── RUN ────────────────────────────────────────────────
result = graph.invoke({
    "question": "Compare Tesla vs BYD stock performance and future outlook",
    "subtasks": [],
    "retrieved_documents": [],
    "draft_report": "",
    "critique": "",
    "iteration_count": 0
})

print(f"\n{'='*60}")
print(" FINAL REPORT:")
print('='*60)
print(result['draft_report'])
print(f"\nBased on {len(result['retrieved_documents'])} verified documents")