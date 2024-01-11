# LangChain RAG Examples

This repo contains two examples (in Python-form and Jupyter Notebook form) of Retrieval-Augmented Generation (RAG) with LangChain.
1. **[Easy RAG with LangChain and OpenAI (notebook)](openai_langchain_notebook.ipynb)**
2. **[Local RAG with LangChain and a Norwegian LLM (notebook)](local_langchain_notebook.ipynb)**

## Setup
Install Python 3.10 and the required frameworks from `required.txt`.

### openai_langchain.py
Add text files into the folder `lang_chain_docs_2024`. Edit the `.env.example` and add your OpenAI API key. Now, test a query.

### local_langchain.py
Add your own pdf in the `data` folder. The pdf should be written in Norwegian. Query away.