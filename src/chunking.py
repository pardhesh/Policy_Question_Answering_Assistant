from typing import List, Dict

def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text.strip(),
                "source": source
            })

            start += chunk_size - overlap

    return chunks
