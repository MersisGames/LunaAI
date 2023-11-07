import os
import openai
import pandas as pd
import uuid
import numpy as np
import time
import constants
import requests

from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#CONSTANTS
LONG_TERM_MEMORY_FILE = "long_term_memory.txt" 
NEW_DATA_FILE = 'newData.csv'
GPT_MODEL = 'gpt-4-1106-preview'
EMBEDING_ENGINE = "text-embedding-ada-002"

#VARIABLES
SHORT_TERM_MEMORY_FILE = str(uuid.uuid4()) + "_STM.txt"
data_results = None
f_response = ""


# Generate a unique file name based on UUID

#INIT
openai.api_key = constants.APIKEY 
paragraphs = pd.DataFrame(columns=["text", "Embedding"])




# Load data if it doesn't exist
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


#TODO: Check why we have two identical functions with different names
def save_to_long_term_memory(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def save_file(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def delete_file(file):
    if os.path.exists(file):
        os.remove(file)


def search(query, data, num_results=5):
    query_embed = get_embedding(query, engine=EMBEDING_ENGINE)
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embed))
    data = data.sort_values("Similarity", ascending=False)
    data_results = data.iloc[:num_results][["text"]]
    
    #Prep Messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."
        },
        {
            "role": "user",
            "content": f"I have this question: {query}, and you have this data to help you: {data_results} to generate a response to that question. Please start your answer continuing this first part of answer {f_response} please answer with an alternative data with different wording and dont forget what you chat earlier, here the memory chat {SHORT_TERM_MEMORY_FILE}."
        }
    ]

    #Generate Response
    start_time = time.time() 
    full_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        max_tokens=100,
    ).choices[0].message["content"]
    
   # Split the text into sentences using ". " as the delimiter
    sentences = full_response.split(". ")

    # Join all sentences starting from the second one
    LTM_response = ". ".join(sentences[1:])
    
    end_time = time.time()  # Marcar el tiempo de finalización
    elapsed_time = end_time - start_time

    save_file(query, LTM_response, SHORT_TERM_MEMORY_FILE)
    save_to_long_term_memory(query, LTM_response, LONG_TERM_MEMORY_FILE)
    

    print(f" {LTM_response} [{elapsed_time:.2f} ]")
    return LTM_response

def generate_response_LTM(question, SHORT_TERM_MEMORY_FILE):
    try:
        # Check short term memory exists.
        if os.path.exists(SHORT_TERM_MEMORY_FILE):
            with open(SHORT_TERM_MEMORY_FILE, 'r') as stm_file:
                short_term_memory = stm_file.read()
        else:
            short_term_memory = ""
        

        with open(LONG_TERM_MEMORY_FILE, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    saved_question = lines[i].strip()
                    saved_response = lines[i + 1].strip()
                    if saved_question and saved_response:
                        saved_question_embedding = get_embedding(saved_question, engine="text-embedding-ada-002")
                        current_question_embedding = get_embedding(question, engine="text-embedding-ada-002")
                        similarity = cosine_similarity(saved_question_embedding, current_question_embedding)
                        if similarity > 0.9:
                            response = saved_response
                            print(response)

                            return response
    except FileNotFoundError:
        pass

    # Add short term memory as context to conversation.
    question = f"I have this question: {question}, and you have this data to help you: {data_results} to generate a response to that question. Please answer with an alternative data with different wording and don't forget what you chatted earlier. Here's the memory chat:\n{short_term_memory}"

    response = search(question, paragraphs)
    return response

def generate_short_response(question): 
    short_m = [
        {
            "role": "system",
            "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."
        },
        {
            "role": "user",
            "content": f"I have this question: {question}, give a response in no more than a line being conversational. Dont forget what you chat earlier, here the memory chat {SHORT_TERM_MEMORY_FILE}."
        }
    ]

    start_time = time.time()
    
    # Create a conversation with ChatGPT
    quick_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=short_m,
        max_tokens=200,
    ).choices[0].message["content"]
    sentences = quick_response.split(". ")
    first_response = sentences[0]
    end_time = time.time()  # Marcar el tiempo de finalización
    elapsed_time = end_time - start_time
    
    print(first_response)
    return first_response
    


while __name__ == "__main__":
    question = input("Question: ")
    if question.lower() == "exit":
        delete_file(SHORT_TERM_MEMORY_FILE)
        break
    generate_short_response(question)
    response = generate_response_LTM(question, SHORT_TERM_MEMORY_FILE)
