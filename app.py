
from flask import Flask, render_template, request
from functools import lru_cache
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()


@lru_cache(maxsize=1)
def get_rag_chain():
    """Initialize the RAG pipeline on the first chat request."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key or not openai_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY and OPENAI_API_KEY must be configured."
        )

    # Initialize embeddings only when the first question arrives.
    embeddings = download_hugging_face_embeddings()

    index_name = "medicalbot"

    docsearch = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.4,
    max_tokens=300,
    api_key=openai_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(
        retriever,
        question_answer_chain
    )


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return "Medical RAG Chatbot is running.", 200

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question.", 400

    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])

    except Exception:
        app.logger.exception("Chat request failed")
        return "The chatbot is temporarily unavailable. Please try again.", 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False
    )
