import gradio as gr
from utils.whisper import transcribe_audio
from agents.langchain_agent import agent_with_memory
from langsmith import traceable

def chat_and_update_history(message, history):
    response = agent_with_memory.run(message)
    updated = f"{history}\n🧑 {message}\n🤖 {response}" if history else f"🧑 {message}\n🤖 {response}"
    return updated, ""  # Updated chat + clear input

@traceable(name="Eleo_Gradio_Text")
def chat_with_agent(text):
    return agent_with_memory.run(text)

@traceable(name="Eleo_Gradio_Audio")
def handle_audio(audio_file):
    transcript = transcribe_audio(audio_file)
    response = agent_with_memory.run(transcript)
    return f"Transcription:\n{transcript}\n\nAnswer:\n{response}"

# ✅ UI
with gr.Blocks() as interface:
    gr.Markdown("# Hallochen! I'm Eleo – German QA Bot")
    gr.Markdown("Ask for everything you want to know to learn German efficiently!")

    with gr.Tab("💬 Text Chat"):
        chat_history = gr.Textbox(label="Conversation", interactive=False, lines=20, show_copy_button=True)
        txt_input = gr.Textbox(placeholder="Ask something…", lines=1)
        txt_btn = gr.Button("Ask")

        txt_input.submit(chat_and_update_history, inputs=[txt_input, chat_history], outputs=[chat_history, txt_input])
        txt_btn.click(chat_and_update_history, inputs=[txt_input, chat_history], outputs=[chat_history, txt_input])

    with gr.Tab("🎙 Upload Audio"):
        audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload .mp3 or .wav")
        audio_output = gr.Textbox(label="Agent Response with Transcript")
        audio_btn = gr.Button("Transcribe & Ask")
        audio_btn.click(handle_audio, inputs=audio_input, outputs=audio_output)

if __name__ == "__main__":
    interface.launch()

