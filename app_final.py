import gradio as gr
from utils.whisper_utils import transcribe_audio
from agents.langchain_agent import agent_with_memory
from langsmith import traceable

def chat_and_update_history(message, history):
    response = agent_with_memory.run(message)
    updated = f"{history}\n🧑 {message}\n🤖 {response}" if history else f"🧑 {message}\n🤖 {response}"
    return updated, "" 

@traceable(name="Eleo_Gradio_Text")
def chat_with_agent(text):
    return agent_with_memory.run(text)

@traceable(name="Eleo_Gradio_Audio")
def handle_audio_and_update(audio_file, history):
    transcript = transcribe_audio(audio_file)
    return chat_and_update_history(transcript, history)

def handle_audio_and_update(audio_file, history):
    if audio_file is None:
        return "❗ Please upload an audio file to transcribe.", history
    try:
        transcript = transcribe_audio(audio_file)
        response = agent_with_memory.run(transcript)
        updated_history = f"{history}\n🧑 {transcript}\n🤖 {response}"
        return response, updated_history
    except Exception as e:
        return f"❌ An error occurred: {str(e)}", history

# UI
with gr.Blocks() as interface:
    gr.Markdown("# Hallochen! I'm Eleo – German QA Bot")
    gr.Markdown("Ask anything to learn German efficiently!")

    chat_history = gr.Textbox(label="Conversation", interactive=False, lines=20, show_copy_button=True)
    txt_input = gr.Textbox(placeholder="Ask something…", lines=1)
    ask_btn = gr.Button("Ask")
    mic_input = gr.Audio(sources="microphone", type="filepath", label="", show_label=False)
    mic_btn = gr.Button("🎤")

    # Text input events
    txt_input.submit(chat_and_update_history, inputs=[txt_input, chat_history], outputs=[chat_history, txt_input])
    ask_btn.click(chat_and_update_history, inputs=[txt_input, chat_history], outputs=[chat_history, txt_input])

    # Voice input event
    mic_btn.click(handle_audio_and_update, inputs=[mic_input, chat_history], outputs=[chat_history, txt_input])

if __name__ == "__main__":
    interface.launch()

