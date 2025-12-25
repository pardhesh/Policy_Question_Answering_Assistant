from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

def main():
    load_dotenv()
    print("🔹 Policy Question Answering Assistant")
    print("Type 'exit' to quit.\n") 

    rag = RAGPipeline()

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        answer = rag.answer_question(question)
        print("\n" + answer + "\n")

if __name__ == "__main__":
    main()
