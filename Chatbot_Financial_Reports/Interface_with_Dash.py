pip install --upgrade --quiet langchain langchain-openai chromadb beautifulsoup4
pip install pypdf
pip install dash
pip install dash-bootstrap-components

import dash
from dash import Dash, dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.layout = dbc.Container([
    dbc.Row([
        dbc.Card([
            dbc.CardHeader("Know your competitors. Chatbot", style={'fontSize': '27px', 'width': '120vh'}),
            dbc.CardBody([dcc.Markdown(id='dataname', style={'whiteSpace': 'pre-line'})])
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Store(id='store2', data=[]),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                    , dcc.Store(id='store', data=[])
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px 0'
                }
            ),
            dbc.Card([
                dbc.CardHeader("Earnings Report Summary"),
                dbc.CardBody([dcc.Markdown(id='text', style={'whiteSpace': 'pre-line'})])
            ]),
            dbc.Card([
                dbc.CardHeader("Risks and Opportunities"),
                dbc.CardBody([dcc.Markdown(id='text2', style={'whiteSpace': 'pre-line'})])
            ]),
            dbc.Card([
                dbc.CardHeader("Ask a Question Concerning the uploaded file"),
                dbc.CardBody([dcc.Input(id='input', debounce=True, style={'width': '110vh', 'height': '10vh'})])
            ]),
            dbc.Card([
                dbc.CardHeader("Response"),
                dbc.CardBody([dcc.Markdown(id="text3", style={'whiteSpace': 'pre-line'})])
            ]),
        ], width=8),
    ], className="mt-4"),
])

@callback(
    Output('text3', 'children'),
    Input('input','value')
)

def update_layout(question_asked):
    if question_asked:
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
        display_answer = chain.invoke(question_asked)
       

        return display_answer

    else:
        return no_update

if __name__ == '__main__':
    app.run_server(port=780, debug=True)
