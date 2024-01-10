from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


print("Creating loader...")
loader = DirectoryLoader("./lang_chain_docs_2024", glob="**/*.txt", show_progress=True, recursive=True,
                         silent_errors=True, loader_cls=TextLoader)

print("Loading documents...")
documents = loader.load()
print("First document:", documents[0])

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

print("Creating vectorstore...")
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

print("Creating retriever...")
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

print("Connecting to OpenAI LLM...")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print("Invoking RAG...")
result = rag_chain.invoke("In LangChain, how can I get the graph of a chain and print it as ascii?")
print(result)

rag_chain.get_graph().print_ascii()

vectorstore.delete_collection()
