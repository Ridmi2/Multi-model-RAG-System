from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import json
import os 

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

#State 
class FinancialResearchState(TypedDict):
    question: str               #user question
    subtasks: list              #planner output
    retrieved_documents: list   #researcher output
    draft_report: str           #synthesizer output
    critique: str               #critic output
    iteration_count: int        #how many retry loops

#Planner node
def planner_node(state: FinancialResearchState) -> dict:
    print("Planner Agent Running...") 

    question = state['question']

    response = llm.invoke(f"""You are financial research planner.
Break this question intpo 3-4 specific research subtasks.
Return ONLY a JSON arrray. No markdown, no code fences, no
explanation.
Start your response with [and end with]                          

Question: {question}""")    

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [response.content]

    print(f"Subtasks creaed: {len(subtasks)}")
    for i, task in enumerate(subtasks):
        print(f" {i+1}.{task}")  

    return {"subtasks": subtasks, "iteration_count":0}  

#Build Graph
graph_builder = StateGraph(FinancialResearchState)
graph_builder.add_node("planner",planner_node)
graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner",END)
graph = graph_builder.compile()

# Test
result = graph.invoke({
    "question": "Compare Tesla vs BYD stock performance and future outlook",
    "subtasks": [],
    "retrieved_documents": [],
    "draft_report": "",
    "critique": "",
    "iteration_count": 0
    })

print(f"\n Planner complete..")
print(f"Total subtasks: {len(result['subtasks'])}")
