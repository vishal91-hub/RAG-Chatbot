# RAG-Chatbot
# 🤖 HR Policy RAG Chatbot using Amazon Bedrock and LangChain

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about HR policies using a public PDF document. It uses Amazon Bedrock's foundation models and embeddings, FAISS as a vector database, and LangChain for orchestration.

---

## 📚 What It Does

1. **Loads** a PDF file (`Leave Policy India`) from a URL.
2. **Splits** the text into chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeds** the text using Amazon Bedrock’s Titan Embedding model.
4. **Stores** the embeddings in a FAISS vector store.
5. **Indexes** the document for semantic search.
6. **Queries** the index using Claude (via Bedrock) for accurate, context-aware responses.

---
