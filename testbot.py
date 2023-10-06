import os
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import uuid

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

# Extract the page_content from each text and convert it into a DataFrame
texts = [str(i.page_content) for i in texts]  # List of paragraphs
paragraphs = pd.DataFrame(texts, columns=["text"])

paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
paragraphs.to_csv('newData.csv')

def save_to_file(question, response, file):
    with open(file, 'a') as f:
        f.write(f'Question: {question}\nResponse: {response}\n\n')

def delete_file_if_exists(file):
    if os.path.exists(file):
        os.remove(file)

def save_to_long_term_memory(question, response, file):
    with open(file, 'a') as f:
        f.write(f'Pregunta: {question}\nRespuesta: {response}\n\n')

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
            "content": f"I have this question: {query}, and you have this data to help you: {datafinder} to generate a response to that question. please answer with an alternative data with different wording."
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

import numpy as np

def generate_response_LTM(question):
    try:
        with open(long_term_memory, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                saved_question = lines[i].strip()
                saved_response = lines[i + 1].strip()
                saved_question_embedding = get_embedding(saved_question, engine="text-embedding-ada-002")
                current_question_embedding = get_embedding(question, engine="text-embedding-ada-002")
                similarity = cosine_similarity(saved_question_embedding, current_question_embedding)
                if similarity > 0.7:
                    # Usar la respuesta de la pregunta similar
                    response = saved_response
                    print(response)
                    return response
    except FileNotFoundError:
        pass  # Si el archivo no existe, continúa con la búsqueda

    # Si no hay una pregunta similar en la memoria a largo plazo, continuar con la búsqueda
    search(question, paragraphs)

while True:
    question = input("Question: ")
    if question.lower() == "exit":
        delete_file_if_exists(short_term_memory_file)
        break
    response = generate_response_LTM(question)
