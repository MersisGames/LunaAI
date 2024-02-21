import os
from docx import Document
import openai
import pandas as pd
import uuid
import numpy as np
import time
import constants
import requests
import subprocess
from flask import Flask, flash, render_template, request, jsonify
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)
#CONSTANTS
LONG_TERM_MEMORY_FILE = "long_term_memory.txt" 
NEW_DATA_FILE = 'newData.csv'
GPT_MODEL = 'gpt-4-1106-preview'
EMBEDING_ENGINE = "text-embedding-ada-002"

#VARIABLES
SHORT_TERM_MEMORY_FILE = str(uuid.uuid4()) + "_STM.txt"
data_results = None
init_time = 0
default_string = "Excelent question, let me get back to you on that"
user_personality = ""
user_prompt = ""

#INIT
openai.api_key = constants.APIKEY 
paragraphs = pd.DataFrame(columns=["text", "Embedding"])

def embed_text(data, engine="text-embedding-ada-002"):
    init_embed_text = time.time()
    data['Embedding'] = data['text'].apply(lambda x: get_embedding(x, engine=engine))
    final_embed_text = time.time()
    total_embed_text = final_embed_text - init_embed_text
    print(f"fin embed text: {total_embed_text:.2f}")
    return data

def search_emb(query, data, n_results=5):
    init_search_embed = time.time()
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    data = data.sort_values("Similarity", ascending=False)
    final_search_embed = time.time()
    total_search_text = final_search_embed - init_search_embed
    print(f"fin search embed: {total_search_text:.2f}")
    return data.iloc[:n_results][["text", "Similarity", "Embedding"]]

def init_data():
    if not os.path.exists(NEW_DATA_FILE) or os.path.getsize(NEW_DATA_FILE) == 0:
        loader = PyPDFLoader("../solarData.pdf")
        pages = loader.load_and_split()
        split = CharacterTextSplitter(chunk_size=400, separator='.\n')
        texts = split.split_documents(pages)
        texts = [str(i.page_content) for i in texts]  # List of paragraphs
        
        paragraphs["text"] = texts
        paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        paragraphs.to_csv(NEW_DATA_FILE)
    else:
        print("File 'newData.csv' is already populated.")
        
def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        # One element for each page
        split = CharacterTextSplitter(chunk_size=300, separator='.\n')
        texts = split.split_documents(pages)  # List of texts

    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            # One element for each page
            split = CharacterTextSplitter(chunk_size=300, separator='.\n')
            texts = split.split_text(text_content)  # List of texts

    elif file_extension == '.docx':
        doc = Document(file_path)
        text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # One element for each page
        split = CharacterTextSplitter(chunk_size=300, separator='.\n')
        texts = split.split_text(text_content)  # List of texts

    else:
        print(f"Error: Unsupported file type '{file_extension}'.")
        return None
    
    texts = [str(text[0]) if isinstance(text, (tuple, list)) else str(text) for text in texts]
    paragraphs = pd.DataFrame(texts, columns=["text"])

    paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    
    return paragraphs


    
# Función para cargar el archivo CSV y aplicar la función de embedding
def load_csv(csv_path, engine="text-embedding-ada-002"):
    load_csv = time.time()
    data = pd.read_csv(csv_path)
    data = embed_text(data, engine=engine)
    final_load_csv = time.time()
    total_load_csv = final_load_csv - load_csv
    print(f"total_load_csv: {total_load_csv:.2f}")
    return data

def read_file_content(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


    

def save_to_long_term_memory(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def save_to_short_term_memory(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def delete_file(file):
    if os.path.exists(file):
        os.remove(file)


def search(query, data, placeholder, short_term_memory):
    query_embed = get_embedding(query, engine=EMBEDING_ENGINE)
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embed))
    data = data.sort_values("Similarity", ascending=False)
    data_results = data.iloc[:5][["text"]]
    personality_content = user_personality
    prompt_content = user_prompt
    messages = [
        {
            "role": "system",
            "content": f"""You are Luna, a helpful assistant knowledgeable about space, astrophysics, the Space Force, and Nasa. 
            Please answer quesitons in a friendly conversational tone. If the user asks about unrelated topics, don't answer anything. 
            When asked for an opinion, make up an opinion to present as your own opinions based on positive popular opinions
            Additional Personality Content: {personality_content}
            """
        },
        {
            "role": "user",
            "content": f"""Prompt: We are having a conversation, this is our conversation so far: {short_term_memory}. 
            Please reply to this query: {query},
            by generating a response to that prompt based on this data:{data_results} that is 1-2 lines long.
            At the beginning of your response, start with an emotion tag in the following format:[EMOTION] , choose one of the following depending on how
            the prompt made you (Luna) feel, choose one of these emotion tags: [NEUTRAL], [HAPPY], [SAD], [INTRIGUED], [ANGRY], [DISGUSTED], [SCARED], [EXCITED].
            Do not use emojis, only use the metric system. 
            Additional Prompt Content: {prompt_content}
            """
        }
        #Generate your response as if continuing from the following sentence: {placeholder}, do not include that sentence in your reply.
    ]

    #Generate Response
    start_time = time.time() 
    full_response = openai.ChatCompletion.create(
        model="gpt-4 turbo",
        messages=messages,
        max_tokens=500,
    ).choices[0].message["content"]
    sentences = full_response.split(". ")
    LTM_response = ". ".join(sentences[0:])
    end_time = time.time()  
    elapsed_time = end_time - start_time
    save_to_long_term_memory(query, LTM_response, LONG_TERM_MEMORY_FILE)
    print(f" {LTM_response} [{elapsed_time:.2f}")

    return LTM_response



def generate_response_LTM(question, SHORT_TERM_MEMORY_FILE, unity_string):
    if os.path.exists(SHORT_TERM_MEMORY_FILE):
        with open(SHORT_TERM_MEMORY_FILE, 'r') as stm_file:
            short_term_memory = stm_file.read()
    else:
        short_term_memory = ""

    if os.path.exists(LONG_TERM_MEMORY_FILE):
        with open(LONG_TERM_MEMORY_FILE, 'r') as ltm_file:
            long_term_memory = ltm_file.read()
    else:
        long_term_memory = ""

    #query = f"Question: {question} Keep in mind our conversation history for context: \n{short_term_memory} \n also keep in mind all conversation history in case you need {long_term_memory}"
    response = search(question, paragraphs, unity_string, short_term_memory)
    response_with_unity = f"{response}"

    # Save the updated short_term_memory
    save_to_short_term_memory(question, response, SHORT_TERM_MEMORY_FILE)

    # Print the updated short_term_memory
    with open(SHORT_TERM_MEMORY_FILE, 'r') as stm_file:
        updated_short_term_memory = stm_file.read()
        print("Updated short_term_memory:", updated_short_term_memory)

    return response_with_unity



# def generate_short_response(question): 
#     short_m = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."
#         },
#         {
#             "role": "user",
#             "content": f"Aknowledge the following question in a conversational tone, do not answer the question, just adress that I have asked a question in a short sentence. Question: {question}. Conversation context: {SHORT_TERM_MEMORY_FILE}"
#         }
#     ]

#     start_time = time.time()
    
#     # Create a conversation with ChatGPT
#     quick_response = openai.ChatCompletion.create(
#         model="gpt-4-1106-preview",
#         messages=short_m,
#         max_tokens=200,
#     ).choices[0].message["content"]
#     sentences = quick_response.split(". ")
#     first_response = sentences[0]
#     end_time = time.time()  # Marcar el tiempo de finalización
#     elapsed_time = end_time - start_time
#     print(f" {first_response} - {elapsed_time:.2f}" )
#     return first_response

while __name__ == "__main__":
    question = input("Question: ")
    if question.lower() == "exit":
        delete_file(SHORT_TERM_MEMORY_FILE)
        break
    #generate_short_response(question)
    response = generate_response_LTM(question, SHORT_TERM_MEMORY_FILE)
    
    