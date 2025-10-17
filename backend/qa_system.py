import json
import os
from urllib.request import urlopen

from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Configuration ---
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY")
ELASTIC_CLOUD_ID = os.environ.get("ELASTIC_CLOUD_ID")
ELASTIC_ENDPOINT_URL = os.environ.get("ELASTIC_ENDPOINT_URL")
elastic_index_name = "gemini-qa"

# --- Document Loading and Indexing ---
def load_and_index_docs():
    """Loads data, splits it into documents, and indexes it in Elasticsearch."""
    if not ELASTIC_API_KEY or (not ELASTIC_CLOUD_ID and not ELASTIC_ENDPOINT_URL):
        print("Skipping QA document indexing because Elastic credentials are not set.")
        return

    # Load documents from a local JSON file
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(file_path, 'r') as f:
        workplace_docs = json.load(f)

    # Split documents into passages
    metadata = []
    content = []
    for doc in workplace_docs:
        content.append(doc["content"])
        metadata.append(
            {
                "name": doc["name"],
                "summary": doc["summary"],
                "rolePermissions": doc["rolePermissions"],
            }
        )

    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    docs = text_splitter.create_documents(content, metadatas=metadata)

    # Index documents into Elasticsearch
    query_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_document"
    )

    if ELASTIC_ENDPOINT_URL:
        es_connection_args = {"es_url": ELASTIC_ENDPOINT_URL, "es_api_key": ELASTIC_API_KEY}
    else:
        es_connection_args = {"es_cloud_id": ELASTIC_CLOUD_ID, "es_api_key": ELASTIC_API_KEY}

    es = ElasticsearchStore.from_documents(
        docs,
        **es_connection_args,
        index_name=elastic_index_name,
        embedding=query_embedding,
    )
    return es

# --- QA Chain ---
def create_qa_chain():
    """Creates a question-answering chain using Elasticsearch and Gemini."""
    if not ELASTIC_API_KEY or (not ELASTIC_CLOUD_ID and not ELASTIC_ENDPOINT_URL):
        return None

    query_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_query"
    )

    if ELASTIC_ENDPOINT_URL:
        es_connection_args = {"es_url": ELASTIC_ENDPOINT_URL, "es_api_key": ELASTIC_API_KEY}
    else:
        es_connection_args = {"es_cloud_id": ELASTIC_CLOUD_ID, "es_api_key": ELASTIC_API_KEY}

    es = ElasticsearchStore(
        **es_connection_args,
        embedding=query_embedding,
        index_name=elastic_index_name,
    )

    retriever = es.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}\n"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    # This part is for testing purposes and will only run when the script is executed directly
    if not ELASTIC_API_KEY or (not ELASTIC_CLOUD_ID and not ELASTIC_ENDPOINT_URL):
        print("Please set the ELASTIC_API_KEY and either ELASTIC_CLOUD_ID or ELASTIC_ENDPOINT_URL environment variables.")
    else:
        print("Loading and indexing documents...")
        load_and_index_docs()
        print("Documents indexed.")
        qa_chain = create_qa_chain()
        question = "what is our sales goals?"
        print(f"Asking: {question}")
        answer = qa_chain.invoke(question)
        print(f"Answer: {answer}")

