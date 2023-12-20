# Import necessary modules
from typing import Dict
from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
from langchain.llms import CTransformers #to get llm
from langchain.prompts import ChatPromptTemplate
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter#splitting text into chunks
from utils.VectorDBStorer import VectorDBStorer

############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
      # TODO Add code here
      pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
      """ Connect to ChromaDB and perform text processing """

      client = chromadb.Client()
      collection_name = "new_scientific_papers"
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
      dataset_name = 'scientific_papers'

      vector_db_storer = VectorDBStorer(client, collection_name, text_splitter, dataset_name)
      collection = vector_db_storer.get_collection()

      """ Define the model for generating responses """

      llm = CTransformers(
            model = "TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            temperature = 0.2
            )

      """ Create a custom prompt template for generating responses """

      custom_prompt_template = """Use the following pieces of information to answer the user’s question.

      Context: {context}
      Question: {question}

      """

      prompt = ChatPromptTemplate.from_template(custom_prompt_template)

      """ Create a pipeline for generating responses """

      chain = prompt | llm

      """ Process each user's question and generate a response """

      output = []

      for text in request.text:
            results = collection.query(
                 query_texts=text,
                 n_results=1)
            context = results['documents'][0][0]
            question = text
            response = chain.invoke({"context": context, "question": question})
            output.append(response)

      """ Return the responses """

      return SchemaUtil.create(SimpleText(), dict(text=output))
