from agents.langchain_agent import agent_with_memory
from utils.whisper_utils import transcribe_audio
import os
# stop to getting out of the scope
def check_query_scope(query: str) -> str | None:
    forbidden_languages = [
        "Portuguese", "Spanish", "French", "Italian", "Russian", "Chinese", "Japanese",
        "Korean", "Arabic", "Hindi", "Polish", "Ukrainian", "Czech", "Slovak", "Hungarian",
        "Romanian", "Bulgarian", "Greek", "Turkish", "Dutch", "Swedish", "Norwegian", "Danish", "Finnish"
    ]
    off_topic_keywords = [
        "aliens", "cake", "stock market", "politics", "bitcoin", "cooking", "football",
        "celebrity", "movies", "books", "makeup", "weather", "fitness"
    ]

    lowered_query = query.lower()

    for lang in forbidden_languages:
        if (
            f"in {lang.lower()}" in lowered_query
            or f"{lang.lower()} word" in lowered_query
            or f"teach {lang.lower()}" in lowered_query
            or f"learn {lang.lower()}" in lowered_query
        ):
            return (
                "I can teach you only German. But many learning tips work for any language: "
                "Stay consistent, speak daily, and talk to others when possible!"
            )

    for keyword in off_topic_keywords:
        if keyword in lowered_query:
            return (
                "I’m here to help you learn German, not talk about other topics. "
                "Want to ask me about grammar, phrases, or conversations in German?"
            )

    return None

def chat_and_update_history(message, history):
    scope_warning = check_query_scope(message)
    if scope_warning:
        bot_reply = scope_warning
    else:
        bot_reply = agent_with_memory.run(message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})
    return history, ""

def handle_audio_and_update(audio_file, history):
    # catch empty/invalid paths or folders
    if not audio_file or not isinstance(audio_file, str) or not os.path.isfile(audio_file):
        return history, None

    try:
        transcript = transcribe_audio(audio_file)
        scope_warning = check_query_scope(transcript)

        response = scope_warning if scope_warning else agent_with_memory.run(transcript)

        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": response})

        return history, None  # Reset mic
    except Exception as e:
        history.append({"role": "user", "content": "[audio]"})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, None
