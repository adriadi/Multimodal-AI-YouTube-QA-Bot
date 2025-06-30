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

def add_chunk_metadata(
    chunks: list,
    video_id: str,
    transcript_title: str,
    author: str,
    tags: list,
    language: str = "en",
    upload_date: str = None,
    source: str = "YouTube"
) -> list:
    from datetime import datetime
    return [
        {
            "content": chunk,
            "metadata": {
                "video_id": video_id,
                "chunk_index": i,
                "transcript_title": transcript_title,
                "author": author,
                "tags": tags,
                "language": language,
                "upload_date": upload_date or str(datetime.today().date()),
                "source": source
            }
        }
        for i, chunk in enumerate(chunks)
    ]

# move chunks to json format
def convert_chunks_to_json(chunks: list) -> list:
    return [
        {
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        }
        for chunk in chunks
    ]
# Saves chunks to a JSON file
def save_chunks_to_json(chunks: list, output_file: str):
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
