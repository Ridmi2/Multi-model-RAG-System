from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os 
import chromadb
import voyageai
import json

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Vector database set up
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

# Planner node 
def planner_node(state: FinancialResearchState) -> dict:
    print("Planner Agent Running ...")
    response = llm.invoke(f"""You are a financial research planner.
Break this question into 3-4 specific research subtasks.
Return ONLY a raw JSON array of simple strings. 
No markdown, no code fences, no objects, no description fields.
Start with [ and end with ]
Example: ["task 1", "task 2", "task 3"]
                          
Question: {state['question']}""")

    try:
        cleaned = response.content.strip()
        subtasks = json.loads(cleaned)
        if subtasks and isinstance(subtasks[0], dict):
            subtasks = [t.get('subtask', str(t)) for t in subtasks]
    except json.JSONDecodeError:
        subtasks = [state['questtion']]

    print(f"{len(subtasks)} subtasks created")
    return {"subtasks": subtasks, "iteration_count":0}

# Researcher Node
SIMILARITY_THRESHOLD = 0.4

def researcher_node(state: FinancialResearchState) -> dict:
    print("\n Researcher Agent Running...")

    all_documents = []
    subtasks = state["subtasks"]

    all_embeddings = voyage_client.embed(
        subtasks, model="voyage-3-lite"
    ).embeddings

    for i, subtask in enumerate(subtasks):
        print(f" Searching for: '{subtask}'")

        #embed the subtask
        query_embedding = all_embeddings[i]

        # Serch vector DB
        results = collection.query(
            query_embeddings = [query_embedding],
            n_results=2,
            include=["documents","distances"]
        )

        docs = results['documents'][0]
        distances = results['distances'][0]

        for doc, distance in zip(docs, distances):
            similarity = (2 - distance) / 2
            if similarity >= SIMILARITY_THRESHOLD:
                all_documents.append({
                    "subtask":subtask,
                    "document": doc,
                    "similarity": round(similarity, 3)
                })
                print(f" Found (similarity={round(similarity, 3)}): {doc[:60]}...")    
            else:    
                print(f" Rejected (similarity={round(similarity, 3)}): {doc[:60]}...")       

    print(f"\n Total documents retrieved: {len(all_documents)}")
    return {"retrieved_documents": all_documents}

# Build Graph
graph_builder = StateGraph(FinancialResearchState)
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("researcher", researcher_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "researcher")
graph_builder.add_edge("researcher",END)

graph = graph_builder.compile()

# Run
result = graph.invoke({
    "question": "Compare Tesla vs BYD stock performance and future outlook",
    "subtasks": [],
    "retrieved_documents": [],
    "draft_report": "",
    "critique": "",
    "iteration_count": 0
})


print(f"\n Research complete..")
print(f"Documents retrieved: {len(result['retrieved_documents'])}")