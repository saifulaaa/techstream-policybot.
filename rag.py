"""Core RAG pipeline for TechStream PolicyBot (LangChain LCEL + FAISS)."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads OPENAI_API_KEY from .env

PROMPT_TEMPLATE = """You are a helpful customer support assistant for TechStream.
Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "That is not in our policy."

Context:
{context}

Question: {question}

Answer:"""


def load_documents(paths):
    """Load .txt and .pdf files into LangChain documents."""
    docs = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
    return docs


def build_vectorstore(paths, chunk_size=300, chunk_overlap=50):
    docs = load_documents(paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(splits, embeddings), len(splits)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vectorstore, k=3):
    """Returns a chain that outputs {"question", "docs", "answer"}."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    answer_chain = (
        (lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    ) | RunnablePassthrough.assign(answer=answer_chain)
