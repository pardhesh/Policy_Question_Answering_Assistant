import os
from typing import List, Dict

def load_documents(data_dir: str) -> List[Dict]:
    documents = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "text": text,
                "source": file_name
            })

    return documents
