from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .config import Config

class DocumentRetriever:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = None
        self.parent_docs: Dict[str, str] = {}

    def build_index(self, raw_text: str):
        """
        Creates parent (large context) and child (precise search) chunks
        and indexes them into a vector store.
        """
        # 1. Create Parent Chunks (approx 1000 chars with overlap)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        parent_texts = parent_splitter.split_text(raw_text)

        # 2. Create Child Chunks (approx 350 chars) mapped to Parent IDs
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

        child_docs: List[Document] = []
        for parent_id, p_text in enumerate(parent_texts):
            p_id_str = f"parent_{parent_id}"
            self.parent_docs[p_id_str] = p_text

            children = child_splitter.split_text(p_text)
            for c_text in children:
                child_docs.append(
                    Document(
                        page_content=c_text,
                        metadata={"parent_id": p_id_str}
                    )
                )

        # 3. Embed and Index Child Chunks
        print(f"Indexing {len(child_docs)} child chunks mapped to {len(parent_texts)} parent blocks...")
        self.vector_store = FAISS.from_documents(child_docs, self.embeddings)

    def retrieve_parent_context(self, query: str, k: int = 1) -> str:
        """
        Searches child vectors and returns the full parent document context.
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been indexed yet.")

        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return ""

        # Retrieve the corresponding parent block
        parent_id = results[0].metadata.get("parent_id")
        return self.parent_docs.get(parent_id, results[0].page_content)