# End-to-End Medical Chatbot using Generative AI (RAG)

## 📌 Project Overview
This project presents an end-to-end medical chatbot designed to provide preliminary healthcare information using modern Natural Language Processing techniques. The system combines document-based semantic retrieval with generative language models to deliver context-aware, reliable, and safe medical responses. It is intended for educational and research purposes only and does not replace professional medical consultation.

---

## 🎯 Problem Statement
Traditional medical chatbots either rely on rigid rule-based systems or unrestricted generative models. Rule-based systems lack scalability and flexibility, while generative models may produce hallucinated or unsafe medical information. This project addresses the need for a document-grounded, reliable medical chatbot that balances accuracy, safety, and scalability.

---

## 🧠 Proposed Solution
The chatbot follows a **Retrieval-Augmented Generation (RAG)** architecture:
- Relevant medical documents are retrieved using semantic similarity.
- Retrieved context is passed to a language model for grounded response generation.
- This approach minimizes hallucinations and improves factual consistency.

---

## 🏗️ System Architecture
1. **Document Ingestion** – Medical PDF documents are loaded and processed.
2. **Text Chunking** – Documents are split into smaller semantic chunks.
3. **Embedding Generation** – Sentence-transformer embeddings are generated.
4. **Vector Storage** – Embeddings are stored in a vector database.
5. **Query Processing** – User query is converted into embeddings.
6. **Semantic Retrieval** – Top-k relevant chunks are retrieved.
7. **Response Generation** – A generative AI model produces a context-aware response.

---

## 🛠️ Technologies Used
- **Python**
- **Flask** – Web framework
- **LangChain** – LLM orchestration
- **Pinecone** – Vector database
- **HuggingFace Sentence Transformers** – Embedding model
- **OpenAI GPT-3.5 Turbo** – Generative language model
- **HTML / CSS** – User Interface

---

## 📊 Evaluation Metrics
The system was evaluated based on:
- Response relevance
- Retrieval accuracy
- Response latency
- Task completion rate
- User satisfaction (qualitative)

The document-grounded approach demonstrated improved reliability compared to standalone generative responses.

---

## ⚠️ Ethical Considerations
- The chatbot does **not provide medical diagnosis or prescriptions**.
- Emergency-related queries are handled conservatively.
- Users are advised to consult certified healthcare professionals.
- The system is intended strictly for educational and research use.

---
