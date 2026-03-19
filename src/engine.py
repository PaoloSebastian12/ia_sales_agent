
from dotenv import load_dotenv
import pandas as pd
import os
import shutil
import time
import gc

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import Settings



load_dotenv()

def clean_old_indexes(base_path="./"):
    folders = [f for f in os.listdir(base_path) if f.startswith("merging_index_")]
    folders = sorted(folders, reverse=True)
    for old in folders[2:]:
        try:
            shutil.rmtree(os.path.join(base_path, old))
            print(f"Deleted old index: {old}")
        except Exception as e:
            print(f"Could not delete {old}: {e}")

def build_automerging_index(data_path, save_dir, force_reindex=False):
    if force_reindex:
        save_dir = f"{save_dir}_{int(time.time())}"

    if os.path.exists(save_dir) and not force_reindex:
        storage_context = StorageContext.from_defaults(
            persist_dir=save_dir
        )

        index = load_index_from_storage(storage_context)
    else:
        print("Loading documents...")
        documents = []
        for file in os.listdir(data_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(data_path, file)
                loader = PyMuPDFReader()
                docs = loader.load_data(pdf_path)
                documents.extend(docs)
        print("\nDEBUG TEXT SAMPLE:")
        for d in documents[:2]:
            print(d.text[:500])
            print("="*50)
        node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        print("Parsing nodes...")
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        print("Creating index...")
        print("Total nodes:", len(nodes))
        print("Leaf nodes:", len(leaf_nodes))
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            show_progress=True
        )
        index.storage_context.persist(persist_dir=save_dir)
        clean_old_indexes()
    print("start retreiver...")
    retriever = AutoMergingRetriever(index.as_retriever(
        similarity_top_k=3), storage_context=storage_context, verbose=True)
    print("finally query_engine ...")
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=Settings.llm)
    print("finish auto mergign_ index")
    return query_engine


def build_email_engine(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(
            text=content,
            metadata={"source": "email"}
        )
        documents.append(doc)
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=2)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=Settings.llm)
    return query_engine


def create_sales_agent(technical_query_engine, email_engine, contract_engine):
    technical_tool = QueryEngineTool(
        query_engine=technical_query_engine,
        metadata=ToolMetadata(
            name="technical_manuals",
            description="Use this tool for technical specifications database, AWS overview, and architecture details."
        ),
    )

    email_tool = QueryEngineTool(
        query_engine=email_engine,
        metadata=ToolMetadata(
            name="customer_emails",
            description="Useful for checking previous conversations with customers, their needs, and feedback."
        ),
    )

    contract_tool = QueryEngineTool(
        query_engine=contract_engine,
        metadata=ToolMetadata(
            name="legal_contracts",
            description="Use this for legal terms, Google Cloud Terms of Service (GCP), compliance, and liability questions."
        ),
    )

    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[technical_tool, email_tool, contract_tool],
        verbose=True
    )
    print("usando create_sale_agent")
    return router_query_engine
