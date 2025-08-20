from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from huggingface_hub import InferenceClient
import os


load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Initialize Hugging Face client with token from .env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

def process(state: AgentState) -> AgentState:
    user_message = state['messages'][-1].content   # last user input

    # Use chat.completions API
    response = llm.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=200,
    )

    ai_output = response.choices[0].message["content"]
    print(f"\n AI Response: {ai_output}")


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
while user_input != "Exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Human: ")