import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class KnowledgeBaseHandler:
    """
    Handles the creation, loading, and management of a FAISS-based knowledge base.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = "faiss_index"):
        """
        Initializes the KnowledgeBaseHandler.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            index_path (str): The path to store the FAISS index and documents.
        """
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.index = None
        self.documents = None

    def build_index(self, articles: List[Dict[str, str]], force_rebuild: bool = False):
        """
        Builds or loads a FAISS index from a list of articles.

        Args:
            articles (List[Dict[str, str]]): A list of articles, where each article is a dictionary.
            force_rebuild (bool): If True, forces the rebuilding of the index even if it exists.
        """
        if not force_rebuild and os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}...")
            self.load_index()
            print("Index loaded successfully.")
            return

        print("Building new index...")
        # Combine title and description for embedding
        texts = [f"{article.get('title', '')}: {article.get('description', '')}" for article in articles]
        
        print(f"Encoding {len(texts)} texts into vectors...")
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        
        # Ensure embeddings are float32
        embeddings = np.array(embeddings, dtype='float32')

        # Create a FAISS index
        d = embeddings.shape[1]  # Dimension of vectors
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        
        self.documents = articles
        
        print(f"Index built successfully with {self.index.ntotal} vectors.")
        self.save_index()

    def save_index(self):
        """
        Saves the FAISS index and the documents to disk.
        """
        print(f"Saving index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        # Save documents alongside the index
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        print("Index and documents saved.")

    def load_index(self):
        """
        Loads the FAISS index and documents from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
            
        self.index = faiss.read_index(self.index_path)
        
        with open(f"{self.index_path}.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query_texts: List[str], k: int = 5) -> List[List[Dict[str, any]]]:
        """
        Searches the index for the most similar documents to the query texts.

        Args:
            query_texts (List[str]): A list of query strings.
            k (int): The number of nearest neighbors to retrieve for each query.

        Returns:
            List[List[Dict[str, any]]]: A list of search results for each query.
        """
        if self.index is None:
            print("Index is not built or loaded. Please call build_index() or load_index() first.")
            return []

        query_embeddings = self.model.encode(query_texts, convert_to_tensor=False)
        query_embeddings = np.array(query_embeddings, dtype='float32')
        
        distances, indices = self.index.search(query_embeddings, k)
        
        results = []
        for i in range(len(query_texts)):
            query_results = []
            for j in range(k):
                doc_index = indices[i][j]
                if doc_index != -1:
                    query_results.append({
                        "document": self.documents[doc_index],
                        "distance": distances[i][j]
                    })
            results.append(query_results)
            
        return results

if __name__ == '__main__':
    # Create a handler instance
    kb_handler = KnowledgeBaseHandler(index_path="faiss_test_index")

    # Sample articles for demonstration
    sample_articles = [
        {"title": "Fed Interest Rate Decision", "description": "The US Federal Reserve is expected to raise interest rates by 25 basis points in the next meeting."},
        {"title": "China's Economic Outlook", "description": "China's economy shows signs of slowing down, with GDP growth forecasts being revised downwards."},
        {"title": "European Central Bank Policy", "description": "The ECB maintains its dovish stance, keeping rates unchanged amidst inflation concerns."},
        {"title": "Impact of Oil Prices on Global Economy", "description": "Rising oil prices are fueling inflation worldwide and impacting consumer spending."},
        {"title": "Technological Innovations in Finance", "description": "Fintech companies are disrupting traditional banking with new technologies like blockchain and AI."}
    ]

    # Build the index (force rebuild for demonstration)
    kb_handler.build_index(sample_articles, force_rebuild=True)

    # Verify the index
    print(f"\nIndex contains {kb_handler.index.ntotal} vectors.")

    # --- Example of loading the index ---
    print("\n--- Loading Index Example ---")
    # Create a new handler to simulate a different session
    new_kb_handler = KnowledgeBaseHandler(index_path="faiss_test_index")
    new_kb_handler.load_index()
    print(f"Loaded index contains {new_kb_handler.index.ntotal} vectors.")
    print(f"First document loaded: {new_kb_handler.documents[0]['title']}")

    # --- Example of searching the index ---
    print("\n--- Searching Example ---")
    search_queries = ["US interest rates", "future of finance"]
    search_results = new_kb_handler.search(search_queries, k=2)

    for i, query in enumerate(search_queries):
        print(f"\nResults for query: '{query}'")
        for result in search_results[i]:
            print(f"  - Found: {result['document']['title']} (Distance: {result['distance']:.4f})")
            
    # Clean up the created index files
    if os.path.exists("faiss_test_index"):
        os.remove("faiss_test_index")
    if os.path.exists("faiss_test_index.pkl"):
        os.remove("faiss_test_index.pkl")
    print("\nCleaned up test index files.")