import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_all_data
from src.embedding import load_embedding_model, generate_embeddings
from src.vectorstore import init_vectorstore, add_documents_to_store
from src.search import init_gemini_client
from src.chat_manager import ChatSession


def main():
    # Initialize pipeline
    docs = load_all_data(base_path="data", split=True)
    texts = [d.page_content for d in docs]
    model = load_embedding_model()
    embeddings = generate_embeddings(texts, model)
    _, collection = init_vectorstore()
    add_documents_to_store(collection, docs, embeddings)
    client = init_gemini_client()

    # Start chat session
    chat = ChatSession(collection, model, client)

    print("\n💬 Welcome to SSA Chatbot! Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        answer = chat.ask(query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
