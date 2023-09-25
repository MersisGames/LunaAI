import openai
import wikipediaapi
import constants
import re
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from openai.embeddings_utils import get_embedding
import os

# Configura las claves de la API de OpenAI y Wikipedia (reemplaza con tus propias claves)
openai.api_key = constants.APIKEY

# Configura la Wikipedia API
session = wikipediaapi.Wikipedia("en", extract_format=wikipediaapi.ExtractFormat.WIKI, headers={'User-Agent': 'MiAplicacionLocal/1.0 (Ejecutada localmente)'})

# Definir función para buscar información en Wikipedia
def search_wikipedia(topic):
    page = session.page(topic)
    if page.exists():
        return page.summary
    else:
        return "No se encontró información en Wikipedia sobre " + topic

# Definir función para vectorizar un texto
def vectorize(text):
    return get_embedding(text, engine='text-embedding-ada-002')

# Nombre del archivo CSV
csv_filename = 'lunaData.csv'

# Nombre del archivo de texto para guardar preguntas y respuestas vectorizadas
text_filename = 'data.txt'

# Crear un DataFrame para almacenar las preguntas, respuestas y vectores
qa_data = {
    'Pregunta': [],
    'Respuesta': [],
    'Vector_Pregunta': [],
    'Vector_Respuesta': []
}

# Lista para almacenar preguntas y respuestas vectorizadas como texto
vectorized_text_data = []

while True:
    pregunta = input("Ingresa tu pregunta: ")
    
    if pregunta.lower() == "exit":
        break

    # Utiliza ChatGPT para generar una respuesta rápida
    respuesta_corta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA please be kind and friendly conversational . If the user asks about other topics, just say 'I'm not able to talk about it, please ask questions about our topics'"},
            {"role": "user", "content": pregunta}
        ],
        max_tokens=50
    ).choices[0].message["content"]
    
    # Utiliza ChatGPT para proporcionar la respuesta completa
    respuesta_completa = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA please be kind and friendly conversational . If the user asks about other topics, just say 'I'm not able to talk about it, please ask questions about our topics'"},
            {"role": "user", "content": pregunta}
        ]
    ).choices[0].message["content"]

    # Vectorizar la pregunta y la respuesta
    vector_pregunta = vectorize(pregunta)
    vector_respuesta = vectorize(respuesta_completa)

    # Agregar los datos al diccionario
    qa_data['Pregunta'].append(pregunta)
    qa_data['Respuesta'].append(respuesta_completa)
    qa_data['Vector_Pregunta'].append(vector_pregunta)
    qa_data['Vector_Respuesta'].append(vector_respuesta)

    # Almacenar preguntas y respuestas vectorizadas como texto
    vectorized_text_data.append(f'Pregunta: {vector_pregunta}\nRespuesta: {vector_respuesta}\n')
    
       # Utilizar expresiones regulares para encontrar el primer punto en la respuesta completa
    primer_punto_match = re.search(r'\.', respuesta_completa)

    if primer_punto_match:
        indice_primer_punto = primer_punto_match.end()
        respuesta_corta = respuesta_completa[:indice_primer_punto].strip()
        respuesta_completa_recortada = respuesta_completa[indice_primer_punto:].strip()
        print("Respuesta corta (limitada):")
        print(respuesta_corta)

        # Imprimir la respuesta completa recortada
        print("Respuesta completa:")
        print(respuesta_completa_recortada + " Would you like to ask me something else?")
    else:
        print("No se encontró un punto en la respuesta completa.")


# Crear un DataFrame con los datos
    qa_df = pd.DataFrame(qa_data)

    # Guardar el DataFrame en un archivo CSV en el directorio actual
    csv_path = os.path.join(os.getcwd(), csv_filename)
    qa_df.to_csv(csv_path, index=False)

    # Guardar preguntas y respuestas vectorizadas en el archivo de texto
    text_path = os.path.join(os.getcwd(), text_filename)
    with open(text_path, 'w') as text_file:
        text_file.writelines(vectorized_text_data)

    print("Los datos se han guardado correctamente en 'lunaData.csv' y 'data.txt'.")
