import os
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import Bedrock
data_load=PyPDFLoader('https://www.drishtiias.com/pdf/fundamental-duties.pdf')
data_split=data_load.load_and_split()
print(len(data_split))
print(data_split[0])
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
