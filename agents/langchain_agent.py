import sys
sys.path.append('./')
import os

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from utils.whisper_utils import whisper_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
import json 

# Check if vectorstore exists
project_root = os.path.dirname(os.path.abspath(__file__))
# take ID dynamically from the videos_to_process.json file
entry = json.load(open("videos_to_process.json"))[0]
video_id = entry["url"].split("v=")[-1]
   
vectorstore_path = os.path.abspath(os.path.join(project_root, f"../data/vectorstores/{video_id}_faiss"))


index_path = os.path.join(vectorstore_path, "index.faiss")
if not os.path.exists(index_path):
    raise FileNotFoundError(f"Vectorstore not found at {index_path}. Run chunking + embedding pipeline first.")

db = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# prompt template 
custom_prompt = PromptTemplate.from_template("""
You are Eleo - an assistant that answers questions only using the following transcript chunk.
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
    return_source_documents=True
)

def answer_question_only_result(query: str) -> str:
    return qa_chain.run(query)

qa_tool = Tool(
    name="VideoQA",
    func=answer_question_only_result,
    description="Answer questions about German from the transcribed videos"
)

# Define the agent's personality and identity
system_prompt = """
You are **Eleo**, a helpful and friendly AI tutor helping users learn German through YouTube videos.
Only answer questions related to German language or the provided video transcripts.
If asked about any other topic, respond:
'I can only help with learning German. Some of my general tips may still help with learning any language.'
Always provide context from the videos whenever possible.
Keep answers short, friendly, and focused on language learning.
"""
# Initialize memory + agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.messages.append(system_prompt)

agent_with_memory = initialize_agent(
    tools=[qa_tool, whisper_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

def run_conversation(query: str) -> str:
    return agent_with_memory.run(query)

###TESTING
#if __name__ == "__main__":
    #print("🔍 Testing valid (in-domain) question:")
    #query1 = "Can you explain the difference between 'der' and 'die'?"
    #print(run_conversation(query1))

    #print("\n🚫 Testing invalid (out-of-domain) question:")
    #query2 = "Can you teach me Portuguese?"
    #print(run_conversation(query2))