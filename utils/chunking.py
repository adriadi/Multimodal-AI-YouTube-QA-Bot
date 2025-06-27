from langchain.text_splitter import RecursiveCharacterTextSplitter

# Loads .txt transcript
def load_transcript(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

 # Uses LangChain splitter
def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Adds metadata for each chunk
def add_chunk_metadata(chunks: list, video_id: str, source: str = "YouTube") -> list:
    return [
        {
            "content": chunk,
            "metadata": {
                "video_id": video_id,
                "chunk_index": i,
                "source": source
            }
        }
        for i, chunk in enumerate(chunks)
    ]