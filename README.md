## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
Students and professionals often have to read long PDF documents to find a single answer, which is slow and tiring. A chatbot that answers questions directly from the document would save that time. This experiment builds a question-answering chatbot using RAG (Retrieval-Augmented Generation). It uses an LLM that answers only from the content of a given PDF and remembers earlier questions in the conversation.

### DESIGN STEPS:

#### STEP 1:

Load the PDF and build the knowledge base

Load the PDF with PyPDFLoader, which reads it page by page.
Split the text into small chunks (1000 characters, with an overlap of 150) using RecursiveCharacterTextSplitter.
Convert each chunk into an embedding (a numeric vector) with OpenAIEmbeddings, and store them in a vector database (DocArrayInMemorySearch).

#### STEP 2:

Build the question-answering chain

Create a retriever that finds the 4 chunks most similar to the user’s question.
Create a ConversationalRetrievalChain with the ChatOpenAI model. It sends the retrieved chunks and the question to the LLM, and uses the chat history so follow-up questions make sense.

#### STEP 3:

Run the chat and evaluate

Type questions about the PDF, and the bot answers using only the document’s content.
Ask follow-up questions to check that the chat history works.
Check the source chunks returned with each answer to confirm that the answers come from the PDF.

### PROGRAM:
```python
%pip install -q pypdf docarray

import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

llm_name = "gpt-4o-mini"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def load_db(file, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

import os
print(os.getcwd())       
print(os.listdir("."))

qa = load_db("3m.pdf", "stuff", 4)
chat_history = []

query = "What is this document about?"
result = qa({"question": query, "chat_history": chat_history})
chat_history.extend([(query, result["answer"])])
print(result["answer"])

while True:
    query = input("You (type exit to stop): ")
    if query.lower() == "exit":
        break
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    print("Bot:", result["answer"], "\n")

for doc in result["source_documents"]:
    print(doc.page_content[:4000], "\n---")
```
### OUTPUT:
<img width="1902" height="1031" alt="Screenshot 2026-11-03 230648" src="https://github.com/user-attachments/assets/8c95a446-5454-49b6-91a1-6e77a8facc33" />
<img width="1872" height="1027" alt="Screenshot 2026-11-03 230622" src="https://github.com/user-attachments/assets/30462229-b1d1-43c7-a151-cf4e860a6f0b" />

### RESULT:
A RAG-based question-answering chatbot was designed and implemented using PyPDFLoader, a text splitter, OpenAI embeddings, an in-memory vector store and a ConversationalRetrievalChain with ChatOpenAI. The chatbot answered questions about the uploaded PDF using only its content, and it handled follow-up questions through the stored chat history. Thus, the question-answering chat using RAG was successfully developed and tested.

Tip: in your record, add one or two real questions you asked and a short line from the answers you got, so the result matches your own output.
