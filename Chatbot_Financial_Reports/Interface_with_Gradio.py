pip install --upgrade gradio
pip install openai
pip install --upgrade --quiet langchain langchain-openai chromadb 
pip install pypdf

import gradio as gr
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

openai_api_key = os.getenv('OPENAI_API_KEY', "Your Key here")

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=openai_api_key)

pdfs = ['https://www.annualreports.com/HostedData/AnnualReports/PDF/ASX_CDD_2021.pdf','https://www.annualreports.com/HostedData/AnnualReports/PDF/TSX_ARE_2022.pdf']

all = []
for i in pdfs:
  loader = PyPDFLoader(i)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  all_splits = text_splitter.split_documents(data)
  all.extend(all_splits)

def predict2(message, history):
  history_langchain_format = []
  for human, ai in history:
    history_langchain_format.append(HumanMessage(content=human))
    history_langchain_format.append(AIMessage(content=ai))
  history_langchain_format.append(HumanMessage(content=message))

  vectorstore = Chroma.from_documents(documents=all, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
  retriever = vectorstore.as_retriever(k=4)

  template = """Answer the question based only on the following context:
        {context}
    
        Question: {question}
        """

  prompt = ChatPromptTemplate.from_template(template)

  model = ChatOpenAI(openai_api_key=openai_api_key)

  chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
    )
  
  # Change to chain as stream
  display_answer = chain.invoke(message)
  return display_answer

gr.ChatInterface(predict2,
                chatbot=gr.Chatbot(height=200),
                textbox=gr.Textbox(placeholder="Ask me something about the information provided", container=False, scale=10),
                title="Financial Report Retriever",
                description="Ask, me something about Cardno",
                theme="monochrome",
                examples=["What is the message for shareholder from Cardno", "What were the revenues from Cardno"],
                cache_examples=True,
                retry_btn=None,
                undo_btn="Delete Previous",
                clear_btn="Clear",).launch()
