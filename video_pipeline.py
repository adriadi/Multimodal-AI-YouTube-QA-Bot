from utils.whisper_utils import transcribe_audio
from utils.metadata_extract import extract_youtube_metadata
from utils.chunking import split_into_chunks, add_chunk_metadata
from utils.embedding import convert_to_documents, embed_documents_with_openai, save_vectorstore_faiss
from utils.yt_downloader import download_audio

def process_video(
    video_url: str,
    audio_path: str,
    transcript_path: str,
    vectorstore_path: str,
    model: str = "text-embedding-3-small",
    tags: list = ["German", "grammar"]
):
    # Download audio
    download_audio(video_url, audio_path)

    # Transcribe
    transcript = transcribe_audio(audio_path)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Extract metadata from YouTube
    meta = extract_youtube_metadata(video_url)

    # Chunk + add metadata
    chunks = split_into_chunks(transcript)
    chunk_docs = add_chunk_metadata(
        chunks,
        video_id=meta["video_id"],
        transcript_title=meta["title"],
        author=meta["channel"],
        tags=tags,  
        upload_date=meta["publish_date"]
    )

    # Convert and embed
    documents = convert_to_documents(chunk_docs)
    faiss_index = embed_documents_with_openai(documents, model=model)
    save_vectorstore_faiss(faiss_index, vectorstore_path)

    return f"✅ Successfully processed and stored {len(documents)} documents."
