import sys
sys.path.append('./')

import gradio as gr
from chat_logic import chat_and_update_history, handle_audio_and_update

def create_interface():
    with gr.Blocks(css="css/theme.css") as interface:
        gr.HTML("""
        <style>
/* App background */
.gradio-container {
  background: radial-gradient(circle at center, #f8f9fb, #e8ecf1);
  font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
  padding: 48px;
  box-sizing: border-box;
}

/* Title */
#component-0 {
  font-size: 1.8rem;
  font-weight: 700;
  color: #1e293b;
  text-align: center;
  margin-bottom: 1.5em;
}

/* All content boxes */
#chatbot, #user-input, #mic-input {
  background: #ffffff;
  border-radius: 24px;
  padding: 24px;
  box-shadow:
    8px 8px 16px rgba(209, 217, 230, 0.6),
    -8px -8px 16px #ffffff;
  color: #1e293b;
  font-weight: 500;
}

/* Minimum height only for chatbot */
#chatbot {
  min-height: 300px;
}

/* Row styling */
.gr-row {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-top: 24px;
  flex-wrap: wrap;
}

/* Textbox input */
textarea, input[type="text"] {
  background: #ffffff;
  border-radius: 24px;
  border: none;
  padding: 12px 18px;
  font-size: 0.95rem;
  color: #334155;
  box-shadow:
    inset 4px 4px 8px rgba(209, 217, 230, 0.3),
    inset -4px -4px 8px #ffffff;
  width: 100%;
  height: 48px;
  min-width: 260px;
}

/* Audio wrapper box */
.gr-box:has(#mic-input),
.gr-box:has(textarea),
.gr-box:has(input[type="text"]) {
  background: #ffffff !important;
  border-radius: 24px !important;
  box-shadow:
    8px 8px 16px rgba(209, 217, 230, 0.4),
    -8px -8px 16px #ffffff;
  padding: 16px !important;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
}

/* Inner audio style */
#mic-input {
  background: transparent !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  width: 100%;
}

/* Send button */
#send-btn {
  background: #fef6e4 !important;
  color: #1e293b !important;
  border-radius: 24px !important;
  padding: 12px 20px !important;
  font-weight: 600;
  font-size: 0.95rem;
  box-shadow:
    4px 4px 8px rgba(209, 217, 230, 0.6),
    -4px -4px 8px #ffffff;
  transition: background 0.3s ease;
}

#send-btn:hover {
  background: #fdf2cf !important;
}
        </style>
        """)

        with gr.Row():
            gr.Markdown("## Eleo's Corner - German Learning Assistant")

        chatbot = gr.Chatbot(label="Eleo Chatbot ready to answer!", show_copy_button=False, scale=12, elem_id="chatbot", type="messages")

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask me anything…",
                lines=1,
                scale=6,
                show_label=True,
                label="Your question",
                container=True,
                elem_id="user-input"
            )
            mic_input = gr.Audio(
                sources="microphone",
                type="filepath",
                label="Record",
                show_label=True,
                scale=2,
                container=True,
                elem_id="mic-input"
            )
            send_btn = gr.Button("Send question", variant="primary", scale=1, elem_id="send-btn")

        # Event bindings
        user_input.submit(chat_and_update_history, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
        send_btn.click(chat_and_update_history, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
        mic_input.change(handle_audio_and_update, inputs=[mic_input, chatbot], outputs=[chatbot, user_input])
    
    return interface
