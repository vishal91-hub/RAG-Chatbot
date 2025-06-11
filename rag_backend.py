import os
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import Bedrock

def info_loader():
    data_load=PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
    print(data_load)
    #data_split=data_load.load_and_split()
    data_split=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size=10,
        chunk_overlap=1,
        #length_function=len,
        #is_separator_regex=False
    )

    bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-east-1')
    data_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)

    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index=data_index.from_loaders([data_load])
    print(db_index)
    return db_index

def info_llm():
    session = boto3.Session(profile_name='default')

    BEDROCK_CLIENT = session.client("bedrock-runtime", 'us-east-1')
    llm=Bedrock(
        credentials_profile_name='default',
       model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.1,
        "top_p": 0.9},
            client=BEDROCK_CLIENT)
    return llm

def info_response(index,question):
    rag_llm=info_llm()
    print(index)
    info_rag_query=index.query(question=question,llm=rag_llm)
    print(info_rag_query)
    return info_rag_query
    
