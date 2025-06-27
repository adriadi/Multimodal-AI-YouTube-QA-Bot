import sys
sys.path.append('./')
import os

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from utils.whisper import whisper_tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Check if vectorstore exists
project_root = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.abspath(os.path.join(project_root, "../data/vectorstores/eleo_faiss"))

index_path = os.path.join(vectorstore_path, "index.faiss")
if not os.path.exists(index_path):
    raise FileNotFoundError(f"Vectorstore not found at {index_path}. Run chunking + embedding pipeline first.")

db = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# prompt template 
custom_prompt = PromptTemplate.from_template("""
You are an assistant that answers questions only using the following transcript chunk.
If the answer is not contained within it, reply with "I don't know."

Transcript:
{context}

Question: {question}
Answer:
""")
# build the retrieval QA chain 
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)

def answer_question_only_result(query: str) -> str:
    return qa_chain.invoke({"query": query})["result"]

qa_tool = Tool(
    name="VideoQA",
    func=answer_question_only_result,
    description="Answers questions about the Eleo German learning video"
)

# Initialize memory + agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_with_memory = initialize_agent(
    tools=[qa_tool, whisper_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)