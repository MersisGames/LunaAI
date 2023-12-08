import os
import time
from flask import Flask, flash, render_template, request, jsonify
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import subprocess
from docx import Document

app = Flask(__name__)

user_personality = ""
user_prompt = ""

paragraphs = pd.DataFrame(columns=["text", "Embedding"])
openai.api_key = ""

def embed_text(data, engine="text-embedding-ada-002"):
    data['Embedding'] = data['text'].apply(lambda x: get_embedding(x, engine=engine))
    return data

def search(query, data, n_results=5):
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    data = data.sort_values("Similarity", ascending=False)
    return data.iloc[:n_results][["text", "Similarity", "Embedding"]]


def process_file(file_path, paragraphs):
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

    # Extract the page_content from each text and convert it into a DataFrame
    texts = [str(i.page_content) for i in texts]  # List of paragraphs
    paragraphs = pd.DataFrame(texts, columns=["text"])

    paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    return paragraphs


@app.route('/ask_questions', methods=['GET'])
def ask_questions():
    csv_files_dir = 'csv_files'
    csv_files = [file for file in os.listdir(csv_files_dir) if file.endswith('.csv')]
    return render_template('bot.html', csv_options=csv_files)

paragraphs = pd.DataFrame(columns=["text", "Embedding"])

@app.route('/process_questions', methods=['POST'])
def process_questions():
    global paragraphs
    global user_personality
    global user_prompt

    query = request.form['query']
    csv_selected = request.form['csvSelect']

    csv_file_path = os.path.join('csv_files', f'{csv_selected}')
    paragraphs = load_csv(csv_file_path)

    personality_content = user_personality
    prompt_content = user_prompt

    results = search(query, paragraphs)
    
    for index, row in results.iterrows():
        text_lines = [line.strip() for line in row['text'].split('\n') if line.strip()]
        cleaned_text = ' '.join(text_lines)
        topic = os.path.splitext(csv_selected)[0]
        
        # Construir mensajes con el contenido recibido
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant knowledgeable about {topic}
                Please answer questions in a friendly conversational tone. If the user asks about unrelated topics, don't answer anything. 
                When asked for an opinion, make up an opinion to present as your own opinions based on positive popular opinions. Additional Personality Content: {personality_content}"""
            },
            {
                "role": "user",
                "content": f"""Prompt: Please reply to this query: {query},
                by generating a response to that prompt based on this data:{cleaned_text} that is 1-2 lines long. Additional Prompt Content: {prompt_content}"""
            }
        ]

        # Generar respuesta
        start_time = time.time() 
        full_response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=100,
        ).choices[0].message["content"]
        sentences = full_response.split(". ")
        response = ". ".join(sentences[0:])
        
        return jsonify({'response': response})
    
    return jsonify({'message': 'Resultados impresos en la consola.'})

# Función para cargar el archivo CSV y aplicar la función de embedding
def load_csv(csv_path, engine="text-embedding-ada-002"):
    data = pd.read_csv(csv_path)
    data = embed_text(data, engine=engine)
    return data

def read_file_content(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/')
def index():
    csv_files_dir = 'csv_files'
    csv_files = [file for file in os.listdir(csv_files_dir) if file.endswith('.csv')]

    per_files_dir = 'personality'
    per_files = [file for file in os.listdir(per_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]

    prompt_files_dir = 'prompt'
    prompt_files = [file for file in os.listdir(prompt_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]
    return render_template('bot.html', csv_options=csv_files, per_files=per_files, prompt_files=prompt_files)


@app.route('/process', methods=['POST'])
def process():
    global paragraphs
    file = request.files['file']
    csv_name = request.form['csvName'] 


    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    _, file_extension = os.path.splitext(file.filename)
    unique_filename = f"{csv_name}{file_extension}"
    file_path = os.path.join(uploads_dir, unique_filename)
    file.save(file_path)
    paragraphs = process_file(file_path, paragraphs)

    if paragraphs is not None:
        csv_files_dir = 'csv_files'
        os.makedirs(csv_files_dir, exist_ok=True)
        csv_file_path = os.path.join(csv_files_dir, f'{csv_name}.csv')
        paragraphs.to_csv(csv_file_path, index=False)

        # Renderizar la plantilla con datos
        csv_files = [file for file in os.listdir(csv_files_dir) if file.endswith('.csv')]
        return render_template('bot.html', csv_options=csv_files)

    return 'Error processing the file.'



@app.route('/save_files', methods=['POST'])
def save_files():
    try:
        global user_personality
        global user_prompt

        # Guardar en la carpeta "personality"
        personality_file = request.files['personalityFile']
        save_path_personality = os.path.join('personality', personality_file.filename)
        personality_file.save(save_path_personality)

        # Guardar en la carpeta "prompt"
        prompt_file = request.files['promptFile']
        save_path_prompt = os.path.join('prompt', prompt_file.filename)
        prompt_file.save(save_path_prompt)

        # Leer el contenido de los archivos después de guardarlos
        personality_content = read_file_content(save_path_personality, encoding='utf-8')
        prompt_content = read_file_content(save_path_prompt, encoding='utf-8')

        # Obtener la lista de archivos en las carpetas
        csv_files_dir = 'csv_files'
        csv_files = [file for file in os.listdir(csv_files_dir) if file.endswith('.csv')]

        per_files_dir = 'personality'
        per_files = [file for file in os.listdir(per_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]

        prompt_files_dir = 'prompt'
        prompt_files = [file for file in os.listdir(prompt_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]

        user_personality = personality_content
        user_prompt = prompt_content


        return render_template('bot.html', csv_options=csv_files, per_files=per_files, prompt_files=prompt_files)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/ask_bot/<csv_name>/<query>', methods=['GET'])
def ask_bot_external(csv_name, query):
    global paragraphs

    # Construye la ruta completa al archivo CSV
    csv_file_path = os.path.join('csv_files', f'{csv_name}.csv')

    # Carga el archivo CSV seleccionado
    paragraphs = load_csv(csv_file_path)

    # Obtén el nombre del tema del archivo CSV
    topic = os.path.splitext(csv_name)[0]

    # Usa la función de búsqueda con la pregunta recibida
    results = search(query, paragraphs)

    for index, row in results.iterrows():
        text_lines = [line.strip() for line in row['text'].split('\n') if line.strip()]
        cleaned_text = ' '.join(text_lines)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant knowledgeable about {topic}
                Please answer quesitons in a friendly conversational tone. If the user asks about unrelated topics, don't answer anything. 
                When asked for an opinion, make up an opinion to present as your own opinions based on positive popular opinions"""
            },
            {
                "role": "user",
                "content": f"""Prompt: Please reply to this query: {query},
                by generating a response to that prompt based on this data:{cleaned_text} that is 1-2 lines long.
                """
            }
        ]

        # Genera la respuesta
        start_time = time.time() 
        full_response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=100,
        ).choices[0].message["content"]
        sentences = full_response.split(". ")
        response = ". ".join(sentences[0:])

        return jsonify({'response': response})

    return jsonify({'message': 'Resultados impresos en la consola.'})



if __name__ == '__main__':
    app.run(debug=True)
