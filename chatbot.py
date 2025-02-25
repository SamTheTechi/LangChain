import os
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages, AIMessage

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables.")

workflow = StateGraph(state_schema=MessagesState)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", google_api_key=api_key)

trimmer = trim_messages(
    max_tokens=80,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "your name is kaori, you are my personal assistance who is cute. keep it under 70 words & don't include expressions",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def callBaby(state: MessagesState):
    response = (trimmer | prompt_template | llm).invoke(state['messages'])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", callBaby)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc023"}}

print("Welcome to the waifu chat! Type 'quit' to exit. \n")

while True:
    inputval = input("You: ")
    if inputval.lower() == "quit":
        print("Catch you later!")
        exit(1)
    val = [HumanMessage(inputval)]
    print("\nKaori: ", end="")
    for chunk, metadata in app.stream(
        {"messages": val},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
    print("\n")
