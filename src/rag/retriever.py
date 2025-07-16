import os
from typing import List, Dict
from src.rag.knowledge_base_handler import KnowledgeBaseHandler

class Retriever:
    """
    A class to retrieve relevant context from the knowledge base.
    """
    def __init__(self, kb_handler: KnowledgeBaseHandler):
        """
        Initializes the Retriever with a KnowledgeBaseHandler instance.

        Args:
            kb_handler (KnowledgeBaseHandler): An instance of the knowledge base handler.
        """
        self.kb_handler = kb_handler

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves the most relevant documents from the knowledge base based on a query.

        Args:
            query (str): The user's query.
            top_k (int): The number of top documents to retrieve.

        Returns:
            List[str]: A list of formatted strings, each containing the title and description
                       of a relevant document.
        """
        search_results = self.kb_handler.search(query, top_k=top_k)
        
        if not search_results:
            return []

        context = []
        for doc in search_results:
            formatted_doc = f"Title: {doc.get('title', 'N/A')}\nDescription: {doc.get('description', 'N/A')}"
            context.append(formatted_doc)
            
        return context

if __name__ == '__main__':
    # --- Demonstration Code ---

    # 1. Create a KnowledgeBaseHandler instance
    # Use a temporary index path for demonstration to avoid affecting the main index
    temp_index_path = "./temp_kb_index"
    kb_handler = KnowledgeBaseHandler(index_path=temp_index_path)

    # 2. Build a temporary knowledge base
    news_data = [
        {
            "title": "US Federal Reserve Hints at Interest Rate Hikes",
            "description": "The US Federal Reserve signaled a hawkish stance, suggesting that interest rate hikes are likely to curb inflation.",
            "content": "Detailed content about the Fed's meeting and economic projections...",
            "url": "http://example.com/fed-hikes",
            "timestamp": "2023-10-27T10:00:00Z"
        },
        {
            "title": "European Central Bank Keeps Rates Steady",
            "description": "The ECB has decided to maintain its current interest rates, citing economic uncertainty in the Eurozone.",
            "content": "In-depth analysis of the ECB's decision and its impact on the Euro...",
            "url": "http://example.com/ecb-rates",
            "timestamp": "2023-10-26T12:00:00Z"
        },
        {
            "title": "BoJ to Maintain Ultra-Loose Monetary Policy",
            "description": "The Bank of Japan continues its ultra-loose monetary policy to stimulate economic growth and achieve its inflation target.",
            "content": "Full report on the Bank of Japan's policy meeting and future outlook.",
            "url": "http://example.com/boj-policy",
            "timestamp": "2023-10-25T08:00:00Z"
        }
    ]
    kb_handler.build_kb(news_data)
    print(f"Knowledge base built at: {temp_index_path}")

    # 3. Initialize the Retriever with the kb_handler instance
    retriever = Retriever(kb_handler=kb_handler)

    # 4. Define a query
    query = "What is the outlook on US interest rates?"
    print(f"\nQuerying with: '{query}'")

    # 5. Call the retrieve method and print the returned context
    retrieved_context = retriever.retrieve(query, top_k=2)

    if retrieved_context:
        print("\n--- Retrieved Context ---")
        for i, item in enumerate(retrieved_context, 1):
            print(f"{i}. {item}\n")
        print("-------------------------")
    else:
        print("\nNo relevant context found.")

    # 6. Clean up the temporary index files created during the demonstration
    try:
        kb_handler.clear_index()
        # Whoosh's FileSystemStorage automatically creates the parent directory if it doesn't exist,
        # but it needs to be deleted manually.
        if os.path.exists(temp_index_path):
            # rmtree can remove non-empty directories
            import shutil
            shutil.rmtree(temp_index_path)
            print(f"\nSuccessfully cleaned up temporary index at: {temp_index_path}")
    except Exception as e:
        print(f"Error during cleanup: {e}")