from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
from typing import List, Dict

class ProductRAG:
    def __init__(self, products: List[Dict]):
        self.products = products
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._build_vectorstore()

    def _build_vectorstore(self):
        documents = []
        for product in self.products:
            content = f"""
            Product: {product['name']}
            Price: ${product['price']}
            Category: {product['category']}
            Description: {product['description']}
            Features: {', '.join(product['features'])}
            """
            doc = Document(page_content=content, metadata=product)
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        splits = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def retrieve_products(self, query: str, k: int = 5) -> List[Dict]:
        relevant_docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.metadata for doc in relevant_docs]
