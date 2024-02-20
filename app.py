import openai
from dotenv import load_dotenv
import os
import streamlit as st  
from langchain.text_splitter import CharacterTextSplitter  
from langchain.embeddings.openai import OpenAIEmbeddings  
from langchain.chains.question_answering import load_qa_chain 
from annoy import AnnoyIndex 
from langchain.llms import OpenAI  
import langchain
from bs4 import BeautifulSoup
from datetime import datetime

langchain.verbose = False  
load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=300,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = AnnoyIndex(embeddings.get_embedding_size(), 'euclidean')  # Utiliza AnnoyIndex desde la biblioteca Annoy

    for i, chunk in enumerate(chunks):
        knowledge_base.add_item(i, embeddings(chunk))

    knowledge_base.build(10)  # Puedes ajustar el número de árboles según tus necesidades

    return knowledge_base


def main():
  st.title("PROGRAMA")  

  xml1 = st.file_uploader("Sube tu primer archivo XML", type="xml")  
  xml2 = st.file_uploader("Sube tu segundo archivo XML", type="xml")  
  xml3 = st.file_uploader("Sube tu tercer archivo XML", type="xml")  
  html = st.file_uploader("Sube tu archivo HTML", type="html")  

  text = ""

  for file, parser in [(xml1, 'xml'), (xml2, 'xml'), (xml3, 'xml'), (html, 'html.parser')]:
    if file is not None:
      soup = BeautifulSoup(file, parser)
      text += soup.get_text()

  if text:
    knowledgeBase = process_text(text)
    query = st.text_input('Escribe tu pregunta para los archivos...')
    cancel_button = st.button('Cancelar')

    if cancel_button:
      st.stop()

    if query and knowledgeBase:
      docs = knowledgeBase.similarity_search(query)
      model = "gpt-3.5-turbo-instruct"
      temperature = 0
      llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
      chain = load_qa_chain(llm, chain_type="stuff")

      with get_openai_callback() as cost:
        start_time = datetime.now()
        response = chain.invoke(input={"question": query, "input_documents": docs})
        end_time = datetime.now()
        print(cost)

        st.write(response["output_text"])

if __name__ == "__main__":
    main()  # Llama a la función principal