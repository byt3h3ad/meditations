import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    from llama_index.core import (
        Settings,
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
    )
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import chromadb

    return (
        ChromaVectorStore,
        HuggingFaceEmbedding,
        Settings,
        SimpleDirectoryReader,
        SimpleNodeParser,
        StorageContext,
        VectorStoreIndex,
        chromadb,
        os,
    )


@app.cell
def _():
    import nltk

    nltk.download("punkt_tab")
    nltk.download("stopwords")
    return


@app.cell
def _(os):
    cwd = os.getcwd()
    vector_db_path = cwd + "/vector_db"
    collection_name = "docs_collection"
    return collection_name, vector_db_path


@app.cell
def _(HuggingFaceEmbedding):
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return (embedding_model,)


@app.cell
def _(SimpleDirectoryReader):
    loader = SimpleDirectoryReader(input_files=["meditations.txt"])
    documents = loader.load_data()
    return (documents,)


@app.cell
def _(SimpleNodeParser, documents):
    parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=40)
    nodes = parser.get_nodes_from_documents(documents)
    return (nodes,)


@app.cell
def _(chromadb, collection_name, vector_db_path):
    db = chromadb.PersistentClient(path=vector_db_path)
    chroma_collection = db.get_or_create_collection(name=collection_name)
    return (chroma_collection,)


@app.cell
def _(ChromaVectorStore, StorageContext, chroma_collection):
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return (storage_context,)


@app.cell
def _():
    # The storage_context is a handy wrapper so that we can easily switch the vector_store as and when needed.
    return


@app.cell
def _(VectorStoreIndex, embedding_model, nodes, storage_context):
    from llama_index.core import load_index_from_storage

    try:
        index = load_index_from_storage(storage_context)
    except:
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embedding_model,
        )
    return (index,)


@app.cell
def _():
    from dotenv import load_dotenv

    load_dotenv()
    return


@app.cell
def _():
    from llama_index.llms.groq import Groq

    return (Groq,)


@app.cell
def _(Groq, Settings):
    Settings.llm = Groq("llama-3.1-8b-instant")
    return


@app.cell
def _():
    from llama_index.core.prompts import PromptTemplate

    qa_prompt = PromptTemplate(
        """
    You are a calm meditation guide.

    Speak naturally, like you are guiding someone in thought.

    Use the context below, but do not refer to it explicitly.
    Let the response feel like a natural reflection.

    Context:
    {context_str}

    User reflection:
    {query_str}

    Response:
    """
    )
    return PromptTemplate, qa_prompt


@app.cell
def _(index, qa_prompt):
    query_engine = index.as_query_engine(
        text_qa_template=qa_prompt, similarity_top_k=3
    )
    return (query_engine,)


@app.cell
def _(query_engine):
    response = query_engine.query(
        "How should I think about happiness based on this text?"
    )

    response.response
    return


@app.cell
def _(nodes):
    import random
    import datetime

    random.seed(datetime.date.today().isoformat())

    candidates = random.sample(list(nodes), 3)
    context = max(candidates, key=lambda x: len(x.text)).text
    context
    return (context,)


@app.cell
def _(PromptTemplate):
    daily_prompt = PromptTemplate(
        """
    You are a calm and grounded meditation guide.

    Write a short daily reflection using the passage below.

    Rules:
    - Keep it under 120 words
    - Be gentle, clear, and human
    - Avoid sounding like an explanation or summary
    - End with a soft reflective note

    Passage:
    {context_str}

    Reflection:
    """
    )
    return (daily_prompt,)


@app.cell
def _(Settings, context, daily_prompt):
    _response = Settings.llm.complete(daily_prompt.format(context_str=context))

    _response.text
    return


if __name__ == "__main__":
    app.run()
