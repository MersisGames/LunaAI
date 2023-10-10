import os
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import uuid
import numpy as np

# Generate a unique file name based on UUID
short_term_memory_file = str(uuid.uuid4()) + "_STM.txt"
long_term_memory = "long_term_memory.txt" 

openai.api_key = "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"

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("../solarData.pdf")
datafinder = None
pages = loader.load_and_split()

split = CharacterTextSplitter(chunk_size=400, separator='.\n')

texts = split.split_documents(pages)

# Inicializa paragraphs como un DataFrame vacío
paragraphs = pd.DataFrame(columns=["text", "Embedding"])

if not os.path.exists('newData.csv') or os.path.getsize('newData.csv') == 0:
    texts = [str(i.page_content) for i in texts]  # List of paragraphs
    paragraphs["text"] = texts
    paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    paragraphs.to_csv('newData.csv')
else:
    # El archivo 'newData.csv' ya tiene datos, por lo que no es necesario volver a realizar la extracción
    print("El archivo 'newData.csv' ya contiene datos, no se realizará la extracción nuevamente.")


def save_to_file(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def delete_file_if_exists(file):
    if os.path.exists(file):
        os.remove(file)

def save_to_long_term_memory(question, response, file):
    with open(file, 'a') as f:
        f.write(f'\n\nQuestion: {question}\nResponse: {response}')

def search(query, data, num_results=5):
    global datafinder
    query_embed = get_embedding(query, engine="text-embedding-ada-002")
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embed))
    data = data.sort_values("Similarity", ascending=False)
    final_data = data.iloc[:num_results][["text"]]
    datafinder = final_data
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."
        },
        {
            "role": "user",
            "content": f"I have this question: {query}, and you have this data to help you: {datafinder} to generate a response to that question. please answer with an alternative data with different wording and dont forget what you chat earlier, here the memory chat {short_term_memory_file}."
        }
    ]

    # Create a conversation with ChatGPT
    LTM_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
    ).choices[0].message["content"]

    save_to_file(query, LTM_response, short_term_memory_file)
    save_to_long_term_memory(query, LTM_response, long_term_memory)
    print(LTM_response)

def generate_response_LTM(question, short_term_memory_file):
    try:
        # Verifica si el archivo de memoria a corto plazo existe antes de abrirlo
        if os.path.exists(short_term_memory_file):
            with open(short_term_memory_file, 'r') as stm_file:
                short_term_memory = stm_file.read()
        else:
            short_term_memory = ""
        
        with open(long_term_memory, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    saved_question = lines[i].strip()
                    saved_response = lines[i + 1].strip()
                    if saved_question and saved_response:
                        saved_question_embedding = get_embedding(saved_question, engine="text-embedding-ada-002")
                        current_question_embedding = get_embedding(question, engine="text-embedding-ada-002")
                        similarity = cosine_similarity(saved_question_embedding, current_question_embedding)
                        print(f'Similarity between saved question and current question: {similarity}')
                        if similarity > 0.9:
                            response = saved_response
                            print(response)
                            return response
    except FileNotFoundError:
        pass

    # Agrega el contenido del archivo de memoria a corto plazo al contexto de la conversación
    user_message = f"I have this question: {question}, and you have this data to help you: {datafinder} to generate a response to that question. Please answer with an alternative data with different wording and don't forget what you chatted earlier. Here's the memory chat:\n{short_term_memory}"

    search(user_message, paragraphs)


while True:
    question = input("Question: ")
    if question.lower() == "exit":
        delete_file_if_exists(short_term_memory_file)
        break
    response = generate_response_LTM(question, short_term_memory_file)
