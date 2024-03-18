pip install --upgrade --quiet langchain langchain-openai chromadb beautifulsoup4
pip install pypdf

# Setup
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

openai_api_key = os.getenv('OPENAI_API_KEY', "openAI_KEY_here")

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=openai_api_key)

# Loading Financial Reports in PDF format from a webpage
pdfs = ['https://www.annualreports.com/HostedData/AnnualReports/PDF/ASX_CDD_2021.pdf','https://www.annualreports.com/HostedData/AnnualReports/PDF/TSX_ARE_2022.pdf']

all = []
for i in pdfs:
  loader = PyPDFLoader(i)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  all_splits = text_splitter.split_documents(data)
  all.extend(all_splits)

vectorstore = Chroma.from_documents(documents=all, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = vectorstore.as_retriever(k=4)

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)

query_transformation_chain = query_transform_prompt | chat

query_transformation_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What are the most important points of the reports for both companies?"),
            AIMessage(
                content="Based on the provided context, the most important points of the reports for both companies include the consolidated statement of financial position, consolidated statement of financial performance, consolidated statement of other comprehensive income, consolidated statement of changes in equity, and consolidated statement of cash flows for the year then ended. Additionally, the notes to the consolidated financial statements, which include significant accounting policies and other explanatory information, are also important. The reports also mention the use of non-GAAP financial measures for analysis and evaluation of operating performance, as well as the primary financial statements such as the consolidated balance sheets, consolidated statements of income, consolidated statements of comprehensive income, consolidated statements of changes in equity, and consolidated statements of cash flows."
            ),
            HumanMessage(content="Tell me more!"),
        ],
    }
)

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        # If only one message, then we just pass that message's content to retriever
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

conversational_retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What are the revenues for aecon"),
        ]
    }
)

conversational_retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What are the most important points of the reports for both companies?"),
            AIMessage(
                content="Based on the provided context, the most important points of the reports for both companies include the consolidated statement of financial position, consolidated statement of financial performance, consolidated statement of other comprehensive income, consolidated statement of changes in equity, and consolidated statement of cash flows for the year then ended. Additionally, the notes to the consolidated financial statements, which include significant accounting policies and other explanatory information, are also important. The reports also mention the use of non-GAAP financial measures for analysis and evaluation of operating performance, as well as the primary financial statements such as the consolidated balance sheets, consolidated statements of income, consolidated statements of comprehensive income, consolidated statements of changes in equity, and consolidated statements of cash flows."
            ),
            HumanMessage(content="Tell me more!"),
        ],
    }
)
