from datetime import datetime

from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

set_debug(True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


start_time = datetime.now()

model_id = "RuterNorway/Llama-2-13b-chat-norwegian-GPTQ"
embeddings_model_id = "NbAiLab/nb-bert-large"

print("Loading data...")
loader = PyPDFLoader("./data/my-cv.pdf")
data = loader.load()

print("Splitting data into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24, keep_separator=True)
splits = text_splitter.split_documents(data)

print("Creating vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_id)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("Creating retriever...")
retriever = vectorstore.as_retriever()

print("Creating prompt...")
prompt = PromptTemplate(input_variables=['context', 'question'],
                        template="Du er en assistent for en IT konsulent. "
                                 "Bruk følgende informasjon for å besvare oppgaven. Hvis du ikke vet svaret, "
                                 "så si at du ikke vet det. Svar så presist som mulig."
                                 "\nOppgave: {question}\nInformasjon: {context}\nSvar:")
print("Selected prompt:", prompt)

print("Creating llm...")
llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=llm,
    do_sample=True,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.92,
    repetition_penalty=1.13
)
hf = HuggingFacePipeline(pipeline=pipe)

print("Creating chain...")
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | hf
        | StrOutputParser()
)

print("Invoking chain...")
# prompt = "Nevn alle språk Mikkel kan i en liste."
prompt = "Hvilket programmeringsspråk kan Mikkel best?"
result = rag_chain.invoke(prompt)
print("\n-----------------\n")
print(result)

print("Time used:", datetime.now() - start_time)
