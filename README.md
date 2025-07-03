# 🧠 Eleo – Multimodal AI ChatBot for YouTube Video QA

**Eleo** is an intelligent chatbot that helps users learn German through conversation by analyzing YouTube video content. It leverages **LangChain**, **Whisper**, **FAISS**, and **Gradio** to support both voice and text queries. Built as part of the Ironhack AI final project, Eleo enables users to ask questions about language-learning videos and receive accurate, conversational responses.
![Screenshot 2025-07-03 at 02 18 16](https://github.com/user-attachments/assets/f577e802-f244-40e4-b5a1-da6e413ea962)


---

## 🍉 DEMO

[DEMO deployed to HuggingFace can be found here](https://huggingface.co/spaces/industrialdog/Eleos_German)
[Presentation slides](https://docs.google.com/presentation/d/1_9-4172D3SosNgFNDMg-f9EmXPsP3lrqjV-ebeMxlL4/edit?slide=id.p#slide=id.p)

---

## 🎯 Project Objectives

- Improve accessibility to video content through transcription
- Enable deep question answering over YouTube videos
- Support multimodal user input (voice + text)
- Utilize LangChain Agents with tools and memory
- Log and evaluate interactions with LangSmith
- Deploy a working educational assistant via Gradio

---

## 🛠️ Tech Stack

- **LangChain** (Agents, Memory, RetrievalQA, Tools)
- **OpenAI Whisper** (audio transcription)
- **OpenAI Embeddings + FAISS** (semantic vector DB)
- **Gradio** (UI for text and audio)
- **HuggingFace** (ready for deployment)
---

## 🧱 System Architecture

```
YouTube Videos (videos_to_process.json)
        ↓
video_pipeline.py
⤷ Downloads video audio
⤷ Transcribes via Whisper
        ↓
chunking_embedding.ipynb
⤷ Loads transcripts
⤷ Splits using RecursiveCharacterTextSplitter
⤷ Embeds with OpenAIEmbeddings
⤷ Saves to FAISS vector store
        ↓
langchain_agent.py / langchain_agent.ipynb
⤷ Loads retriever + memory + whisper tool
⤷ Wraps into LangChain Agent with Memory
        ↓
app_final.py (Gradio)
⤷ Handles text and audio input
⤷ Loads UI from ui/app_ui.py
⤷ Chat + mic input using Gradio
⤷ Calls the LangChain agent
⤷ Displays conversational output
```

---

## 📁 File Overview

| File/Folder | Purpose |
|-------------|---------|
| `videos_to_process.json` | List of German-learning videos + tags |
| `video_pipeline.py` | Downloads + transcribes video using Whisper |
| `bach_process.py` | Automates batch processing of multiple videos |
| `chunking_embedding.ipynb` | Chunks and embeds transcripts into FAISS |
| `langchain_agent.py` | Wraps tools + retriever into LangChain Agent |
| `langchain_agent.ipynb` | Interactive notebook version of the agent setup |
| `chat_logic.py` | Handles text/audio → agent interaction |
| `app_final.py` | Gradio interface for text and voice chat |
| `ui/app_ui.py` | Gradio Blocks UI (chat, mic, layout) |
| `utils/whisper_utils.py` | Whisper-based transcription logic |
| `utils/chunking.py` | Transcript loading and splitting |
| `utils/metadata_extract.py` | (Optional) for metadata/tagging enhancements |
| `data/vectorstores/` | Contains FAISS index with embedded chunks |

---

## 🚀 Setup Instructions

### 1. Clone and Setup Environment

```bash
git clone https://github.com/YOUR_USERNAME/Multimodal-AI-YouTube-QA-Bot.git
cd Multimodal-AI-YouTube-QA-Bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=EleoChatBot
```

> Make sure `ffmpeg` is installed for Whisper to run.

---

## ▶️ Run the App

```bash
python app_final.py
```

- Ask questions about german language via **text or voice**
- Real-time transcription + retrieval + response
- Uses **LangSmith traceable decorators** for QA and logging

---

## 📌 Processed Video Topics

Videos come from **Eleo’s Corner**, a YouTube channel focused on German language learning at A1–A2 levels, tagged with:

- Grammar (e.g., articles, vocabulary)
- Listening and conversation practice
- Fast-track German comprehension

---

## 🧠 LangChain Agent Setup

- **Retriever** from FAISS DB
- **Whisper Tool** for audio-based tool use
- **Memory** to preserve conversation flow
- **Parsing resilience** enabled via `handle_parsing_errors=True`

---

## 👩‍💻 Author

**Adrianna Dziadyk** – Final AI Project (Ironhack Berlin, 2025)

---

## 📄 License

This project is licensed for educational use only.
