from typing import List, Dict
from src.search import rag_advanced


class ChatSession:
    """
    Manages multi-turn RAG chat with history context.
    """

    def __init__(
        self, collection, model, client, top_k: int = 3, score_threshold: float = 0.2
    ):
        self.collection = collection
        self.model = model
        self.client = client
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.history: List[Dict[str, str]] = (
            []
        )  # [{role: "user"/"assistant", "content": "text"}]

    def _build_chat_context(self) -> str:
        """
        Combine the last few messages to maintain conversational context.
        """
        recent_history = self.history[-4:]  # keep last 4 turns for efficiency
        return "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history]
        )

    def ask(self, user_query: str) -> str:
        """
        Handle a single user query with context + document retrieval.
        """
        print(f"User: {user_query}")

        # Step 1: Retrieve document-based context
        result = rag_advanced(
            query=user_query,
            collection=self.collection,
            model=self.model,
            client=self.client,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            return_context=True,
        )

        doc_context = result.get("context", "")
        chat_context = self._build_chat_context()

        # Step 2: Build combined prompt
        full_prompt = f"""
You are a helpful assistant answering based on both prior chat history and the given document context.

Chat history:
{chat_context or '(no prior chat yet)'}

Relevant documents:
{doc_context}

User question: {user_query}

Answer as helpfully and concisely as possible, maintaining conversational tone.
"""

        # Step 3: Generate response
        response = self.client.models.generate_content(
            model="models/gemini-2.0-flash",  # Stable, good performance
            contents=full_prompt
        )
        answer = getattr(response, "text", "No answer generated.").strip()

        # Step 4: Store the turn in history
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        print(f"Assistant: {answer[:200]}...")
        return answer