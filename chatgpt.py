import os
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import constants
import vectorizer

openai.api_key = constants.APIKEY

memory_text_filename = 'data.txt'
ST_memory_filename = ''
info_filename = './info.txt'

# Define the data CSV file name
data_filename = 'data.csv'

# Load long-term memory data from the CSV file
try:
    data = pd.read_csv(data_filename)
except FileNotFoundError:
    data = pd.DataFrame(columns=['Question', 'Response', 'Vector_Question', 'Vector_Response'])

# < ---------------------------------------------------------------- >  

# Crear el archivo de memoria a corto plazo al inicio del programa
def create_short_term_memory_file():
    ST_id = str(uuid.uuid4())
    ST_memory_filename = f'ST_memory_{ST_id}.csv'
    
    # Crea el archivo vacío en el sistema de archivos
    with open(ST_memory_filename, 'w', encoding='utf-8'):
        pass  # Esto crea un archivo vacío
    print('Short-term memory created successfully')
    return ST_memory_filename

ST_memory_filename = create_short_term_memory_file()
print('Short-term memory created successfully')

# def vectorize_text(text):
#     return get_embedding(text, engine='text-embedding-ada-002')
# < ---------------------------------------------------------------- >  

# # Función para vectorizar text
# def vectorize_text(text):
#     return get_embedding(text, engine='text-embedding-ada-002')

# # < ---------------------------------------------------------------- >  

# # Cargar y dividir el texto del archivo info.txt en párrafos
# def create_lunaData_file():
#     print('lunaData starting.')
#     with open(info_filename, 'r', encoding='utf-8') as file:
#         text = file.read()
#         print('archivo abierto.')

#     # Dividir el texto en párrafos (ajusta el tamaño de los párrafos según tus necesidades)
#     paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
#     print('dividir parrafos.')

#     # Vectorizar cada párrafo y almacenar en un DataFrame
#     paragraph_embeddings = [vectorize_text(para) for para in paragraphs]  # Vectorizar cada párrafo
#     paragraph_df = pd.DataFrame({'text': paragraphs, 'Embedding': paragraph_embeddings})
#     print('vectorizado.')
#     print(paragraph_df)

#     # Guardar los vectores de párrafos en un archivo CSV (lunaData.csv)
#     file_path = './lunaData.csv'
#     paragraph_df.to_csv(file_path, index=False)
#     print('lunaData Created at:', file_path)
#     print('lunaData Created.')

# create_lunaData_file()

# Cargar los datos de lunaData.csv
luna_data = pd.read_csv('lunaData.csv')

def search_in_long_term_memory(question, data):
    if data.empty:  # Verifica si data está vacío
        return None

    question_vectorized = vectorizer.embed(question)
    data['Similarity'] = data['Vector_Question'].apply(lambda x: cosine_similarity(question_vectorized, x.reshape(1, -1))[0][0])
    data = data.sort_values('Similarity', ascending=False)
    
    # If the maximum similarity is below a threshold, we consider no similar questions found
    similarity_threshold = 0.7  # You can adjust this threshold as needed
    if data.iloc[0]['Similarity'] < similarity_threshold:
        return None
    
    # If we find a similar question, call generate_response_with_long_term_memory
    similar_question = data.iloc[0]['Question']
    similar_response = data.iloc[0]['Response']
    
    # Call generate_response_with_long_term_memory to get an alternative response
    alternative_response = generate_response_with_long_term_memory(question, similar_response)
    
    return similar_question, alternative_response

def generate_response_with_long_term_memory(question, similar_response, data, conversacion_data, ST_memory_filename):
    # Mensajes de conversación con el sistema, la pregunta del usuario y la respuesta existente en la memoria a largo plazo
    mensajes = [
        {"role": "system", "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."},
        {"role": "user", "content": f"I have this question: {question}, and you have this response: {similar_response} to generate a response to that question. Provide me with an alternative response with different wording."}
    ]

    # Crear una conversación con ChatGPT
    LTM_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mensajes,
        max_tokens=50,
    ).choices[0].message["content"]

    # Guardar la pregunta y respuesta en la memoria a largo plazo (data.csv)
    data = data.append({'Question': question, 'Response': LTM_response, 'Vector_Question': vectorizer.embed(question), 'Vector_Response': vectorize_text(LTM_response)}, ignore_index=True)
    data.to_csv(data_filename, index=False)

    # Guardar la pregunta y respuesta en la memoria a corto plazo (ST_memory_UUID.csv)
    conversacion_data['Pregunta'].append(question)
    conversacion_data['Respuesta'].append(LTM_response)
    conversacion_data['Vector_Pregunta'].append(vectorizer.embed(question))
    conversacion_data['Vector_Respuesta'].append(vectorizer.embed(LTM_response))
    conversacion_df = pd.DataFrame(conversacion_data)
    conversacion_df.to_csv(ST_memory_filename, index=False)

    return LTM_response

def create_response_with_lunaData(question, data, max_lines=3):
    question_vectorized = vectorizer.embed(question)  # Vectoriza la pregunta del usuario

    # Convertir las listas de cadenas en vectores en formato numérico
    luna_data['Embedding'] = luna_data['Embedding'].apply(lambda x: eval(x))

    # Calcular la similitud de los vectores entre la pregunta y los vectores en lunaData.csv
    luna_data['Similarity'] = luna_data['Embedding'].apply(lambda x: cosine_similarity([question_vectorized], [x])[0][0])

    # Restaurar el formato de las listas de cadenas para evitar problemas posteriores
    luna_data['Embedding'] = luna_data['Embedding'].apply(lambda x: str(x))

    # Ordenar el DataFrame por similitud en orden descendente
    luna_data_sorted = luna_data.sort_values('Similarity', ascending=False)

    if luna_data_sorted.empty:
        return "I don't have any information related to your question."

    # Obtener hasta las tres líneas de texto más cercanas
    top_lines = luna_data_sorted['text'][:max_lines].tolist()

    # Combinar las líneas en una respuesta única
    luna_response = '\n'.join(top_lines)

    # Restaurar el formato original de Embedding
    luna_data['Embedding'] = luna_data['Embedding'].apply(lambda x: "[" + ", ".join(map(str, x)) + "]")


    mensajes = [
        {"role": "system",
         "content": "You are a helpful assistant knowledgeable about planets, space, astronomy, and NASA called LUNA. Please be kind and friendly in conversation. All your answers must have a conversational and friendly tone. If the user asks about other topics, kindly inform them that you're only able to answer questions about our designated topics."},
        {"role": "user",
         "content": f"I have this question: {question}, and you have this response: {luna_response} to generate a response to that question. Provide me with an alternative response with different wording."}
    ]

    # Crear una conversación con ChatGPT
    data_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mensajes,
    ).choices[0].message["content"]


    data = data.append({'Question': question, 'Response': data_response,
                        'Vector_Question': vectorizer.embed(question),
                        'Vector_Response': vectorizer.embed(data_response)}, ignore_index=True)
    data.to_csv(data_filename, index=False)

    # Guardar la pregunta y respuesta en la memoria a corto plazo (ST_memory_UUID.csv)
    conversacion_data['Pregunta'].append(question)
    conversacion_data['Respuesta'].append(luna_response)
    conversacion_data['Vector_Pregunta'].append(vectorizer.embed(question))
    conversacion_data['Vector_Respuesta'].append(vectorizer.embed(luna_response))
    conversacion_df = pd.DataFrame(conversacion_data)
    conversacion_df.to_csv(ST_memory_filename, index=False)

    return data_response

def save_short_term_memory_and_exit(ST_memory_filename):
    if os.path.exists(ST_memory_filename):
        os.remove(ST_memory_filename)
        print("Short-term memory file deleted.")
    
    print("Goodbye! Session ended.")
    exit()

# Crear una conversación individual (memoria a corto plazo) con un UUID único
conversacion_id = str(uuid.uuid4())
ST_memory_filename = f'lunadata_{conversacion_id}.csv'

# Crear una memoria a corto plazo para esta conversación
conversacion_data = {
    'Pregunta': [],
    'Respuesta': [],
    'Vector_Pregunta': [],
    'Vector_Respuesta': []
}

while True:
    question = input("Enter your question: ")

    if question.lower() == "exit":
        save_short_term_memory_and_exit(ST_memory_filename)

    # Search in long-term memory
    search_result = search_in_long_term_memory(question, data)

    if search_result:
        similar_question, alternative_response = search_result
        print(f"Similar Question found in Long-Term Memory: {similar_question}")
        print(f"Alternative Response generated by Long-Term Memory: {alternative_response}")
    else:
        # If no similar question was found in Long-Term Memory, query lunaData.csv to get an alternative response
        data_response = create_response_with_lunaData(question, data, max_lines=3)
        print(f"Alternative Response generated by lunaData.csv: {data_response}")  

    # Continuar la conversación y generar una respuesta completa
    user_input = input("Continue the conversation: ")
    conversacion_data['Pregunta'].append(question)
    conversacion_data['Respuesta'].append(user_input)
    conversacion_data['Vector_Pregunta'].append(vectorizer.embed(question))
    conversacion_data['Vector_Respuesta'].append(vectorizer.embed(user_input))
