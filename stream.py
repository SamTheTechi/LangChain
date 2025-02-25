import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()


class State(TypedDict):
    questions: str
    context: List[Document]
    answer: str


pc = Pinecone(api_key=os.getenv("PINECONE"))
index = pc.Index("kaori")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv('EMBD'),
    model_name="sentence-transformers/all-mpnet-base-v2")

prompt = hub.pull("rlm/rag-prompt")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_spiltter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
all_spilte = text_spiltter.split_documents(docs)

vector_store.add_documents(documents=all_spilte)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", google_api_key=os.getenv("API_KEY"))


def retrive(state: State):
    docs = vector_store.similarity_search(state["question"])
    return {"context": docs}


def generate(state: State):
    docs = "\n\n".join(doc.page_content for doc in state["context"])
    msg = prompt.invoke({"question": state["question"], "context": docs})
    response = llm.invoke(msg)
    return {"answer": response.content}


grapth_builder = StateGraph(State).add_sequence([retrive, generate])
grapth_builder.add_edge(START, "retrive")
graph = grapth_builder.compile()

result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
