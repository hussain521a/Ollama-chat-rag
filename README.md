# Ollama-chat-rag
Using Ollama, streamlit, and langchain, to create a chatbot with RAG 

## Requirements
- Install Ollama
- Install Python

## Steps

Open command prompt and run these commands seperately

```
ollama pull llama3.1
```
```
ollama pull nomic-embed-text
```
```
pip install streamlit PyPDF2 langchain-community langchain pillow PyMuPDF chromadb
```

Then run this command to start the application

```
streamlit run app.py
```

