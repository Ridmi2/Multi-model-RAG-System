from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

#define state
class ResearchState(TypedDict):
    question : str
    analysis : str 
    report : str
    critique :str

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def analyst_node(state: ResearchState) -> dict:
    print("Analyst node running...")
    question = state["question"]

    response = llm.invoke(f"Analyze this financial question in 2 sentences: {question}")

    return{"analysis": response.content}

def writer_node(state: ResearchState) -> dict:
    print("Writer node running")

    response = llm.invoke(
        f"Based on this analysis: {state['analysis']}\n"
        f"Write a 3 sentence professional report."
    )

    return {"report": response.content}

def critical_node(state: ResearchState) -> dict:
    print("Critic Node Running...")

    response = llm.invoke(
        f"Review this report and list any claims that might be unverified. Be brief. : {state['report']}"
    )

    return {"critique": response.content}


#Build the graph
graph_builder = StateGraph(ResearchState)

#add nodes
graph_builder.add_node("analyst", analyst_node)
graph_builder.add_node("writer", writer_node)
graph_builder.add_node("critic", critical_node)

#Add edges
graph_builder.set_entry_point("analyst")
graph_builder.add_edge("analyst","writer")
graph_builder.add_edge("writer", "critic")
graph_builder.add_edge("critic",END)


#Compile the graph
graph = graph_builder.compile()

#Run 
print("Running Langgraph\n")

result = graph.invoke({
    "question":"How is Tesla performing in the EV market?",
    "analysis": "",
    "report": "",
    "critique":""
})

print(f"Final Report:\n{result['report']}")
print(f"Critique:\n {result['critique']}")