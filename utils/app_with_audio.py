import gradio as gr
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import tempfile
import openai

# Load environment variables
load_dotenv()

# Load FAISS vector store
db = FAISS.load_local(
    folder_path="data/vectorstores/eleo_faiss",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = db.as_retriever()

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=False
)

# Wrap QA chain in a Tool
def ask_question(query: str) -> str:
    return qa_chain.invoke({"query": query})["result"]

qa_tool = Tool(
    name="GermanVideoQA",
    func=ask_question,
    description="Answers questions about learning German based on Eleo's video."
)

# Whisper transcription tool
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file.flush()
        with open(temp_file.name, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript["text"]

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create agent
agent = initialize_agent(
    tools=[qa_tool],
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Gradio function
def handle_input(text_input, audio_input):
    if audio_input is not None:
        text_input = transcribe_audio(audio_input)
    return agent.run(text_input)

# Interface
interface = gr.Interface(
    fn=handle_input,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ask a question..."),
        gr.Audio(source="upload", type="file", label="Upload audio file")
    ],
    outputs="text",
    title="Learn German with Eleo",
    description="Ask a question by typing or uploading your voice. Audio is transcribed using OpenAI Whisper."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
