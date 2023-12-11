import time
import traceback
import uuid
from flask import Flask, render_template, request, jsonify, session
import json
import os
import openai 
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
#from luna_ai import generate_short_response as generate_short_response_bot
from luna_ai import generate_response_LTM as generate_response_LTM_bot, process_file, read_file_content, save_to_long_term_memory, save_to_short_term_memory

app = Flask(__name__)
app.secret_key = os.urandom(24) 

def load_csv(csv_path, engine="text-embedding-ada-002"):
    load_csv = time.time()
    data = pd.read_csv(csv_path)
    data = embed_text(data, engine=engine)
    final_load_csv = time.time()
    total_load_csv = final_load_csv - load_csv
    print(f"total_load_csv: {total_load_csv:.2f}")
    return data


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

@app.before_request
def before_request():
    if 'csv_selected' not in session:
        session['csv_selected'] = None
        
LONG_TERM_MEMORY_FILE = "long_term_memory.txt" 
SUCCESFULL_RESPONSE = "Successfully response generated and saved"
ERROR_RESPONOSE = "No response found"
SHORT_RESPONSE_FILE = 'shortResponse.json'
LONG_RESPONSE_FILE = 'longResponse.json'

response_data = {}
paragraphs = " "
final_paragraphs = " "
csv_file_server = " "
u_string = " "
f_responose = " "
SHORT_TERM_MEMORY_FILE = str(uuid.uuid4()) + "_STM.txt" 


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
    print("paragraphs: ", paragraphs)
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
        global csv_file_server
        global final_paragraphs

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

        print("Personality Content:", personality_content)
        print("Prompt Content:", prompt_content)

        # Obtener la lista de archivos en las carpetas
        csv_files_dir = 'csv_files'
        csv_files = [file for file in os.listdir(csv_files_dir) if file.endswith('.csv')]

        per_files_dir = 'personality'
        per_files = [file for file in os.listdir(per_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]

        prompt_files_dir = 'prompt'
        prompt_files = [file for file in os.listdir(prompt_files_dir) if file.endswith(('.docx', '.txt', '.pdf'))]

        user_personality = personality_content
        user_prompt = prompt_content

        selected_csv = request.form['csvSelect']
        session['csv_selected'] = selected_csv
        csv_file_server = session['csv_selected']
        print('Selected CSV:', csv_file_server)

        # Llamar a la función embed_text después de guardar los archivos
        csv_file_path = os.path.join('csv_files', f'{csv_file_server}')
        paragraphs = load_csv(csv_file_path)
        # Llamar a la función embed_text después de cargar el archivo CSV
        final_paragraphs = embed_text(paragraphs)
        
        print("paragraphs: ", paragraphs)
        # Renderizar la plantilla con datos
        return render_template('bot.html', csv_options=csv_files, per_files=per_files, prompt_files=prompt_files)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/process_questions', methods=['POST'])
def process_questions():
    first_init = time.time()
    global paragraphs
    global user_personality
    global user_prompt
    global final_paragraphs

    query = request.form['query']
    csv_selected = request.form['csvSelect']

    personality_content = user_personality
    prompt_content = user_prompt

    fin_primer_parte = time.time()

    tiempo_transcurrido = fin_primer_parte - first_init
    print(f"FinPrimer parte:  {tiempo_transcurrido:.2f}")
    
    second_init = time.time()
    results = search_emb(query, final_paragraphs)
    fin_2_parte = time.time() 
    tiempo_transcurrido_2do = fin_2_parte - second_init
    print(f"fin segunda parte: {tiempo_transcurrido_2do:.2f}")
    
    for index, row in results.iterrows():
        third_init = time.time()
        text_lines = [line.strip() for line in row['text'].split('\n') if line.strip()]
        cleaned_text = ' '.join(text_lines)
        topic = os.path.splitext(csv_selected)[0]

        # Construir mensajes con el contenido recibido
        messages = [
            {
                "role": "system",
                "content": f"""{personality_content}
                """
            },
            {
                "role": "user",
                "content": f"""Prompt: Please reply to this query: {query},
                by generating a response to that prompt based on this data:{cleaned_text} {prompt_content}
                """
            }
        ]

        # Generar respuesta
        start_time = time.time() 
        full_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
        ).choices[0].message["content"]
        sentences = full_response.split(". ")
        response = ". ".join(sentences[0:])
        fin_3_parte = time.time() 
        tiempo_transcurrido_3er = fin_3_parte - third_init
        print(f"fin tercera parte: {tiempo_transcurrido_3er:.2f}")
        
        save_to_short_term_memory(query, response, SHORT_TERM_MEMORY_FILE)
        save_to_long_term_memory(query, response, LONG_TERM_MEMORY_FILE)
        return jsonify({'response': response})
    
    
    
    return jsonify({'message': 'Resultados impresos en la consola.'})



# def short_response(question):
#     short_response = generate_short_response_bot(question)
#     if os.path.exists(SHORT_RESPONSE_FILE):
#         os.remove(SHORT_RESPONSE_FILE)
#     with open(SHORT_RESPONSE_FILE, 'w') as json_file:
#         json.dump({"response": short_response}, json_file)
#     return jsonify({"message": SUCCESFULL_RESPONSE})

def long_response(u_string, f_responose):
    long_response = u_string ," " , f_responose
    if os.path.exists(LONG_RESPONSE_FILE):
        os.remove(LONG_RESPONSE_FILE)
    with open(LONG_RESPONSE_FILE, 'w') as json_file:
        json.dump({"response": long_response}, json_file)
    return jsonify({"message": SUCCESFULL_RESPONSE})


def clean_json_files(long_response_file):
    try:
        clean_response = {"response": " "}
        
        # if os.path.exists(short_response_file):
        #     with open(short_response_file, 'w') as json_file:
        #         json.dump(clean_response, json_file)
                
        if os.path.exists(long_response_file):
            with open(long_response_file, 'w') as json_file:
                json.dump(clean_response, json_file)
                
        return {"message": "JSON files cleaned successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.route('/ask_question/<unity_string>/<question>', methods=['POST'])
def ask_question(question, unity_string):
    print("Entro a ask question...")
    try:
        clean_json_files(LONG_RESPONSE_FILE)
        print('limpio')
        global paragraphs
        global user_personality
        global user_prompt
        global csv_file_server
        global final_paragraphs
        global u_string
        global f_responose


        personality_content = user_personality
        prompt_content = user_prompt
        print("Personality Content:", personality_content)
        print("Prompt Content:", prompt_content)

        results = search_emb(question, final_paragraphs)

        for index, row in results.iterrows():
            text_lines = [line.strip() for line in row['text'].split('\n') if line.strip()]
            cleaned_text = ' '.join(text_lines)


            # Construir mensajes con el contenido recibido
            messages = [
                {
                    "role": "system",
                    "content": f"""{personality_content}
                    """
                },
                {
                    "role": "user",
                    "content": f"""Prompt: Please reply to this query: {question},
                    by generating a response to that prompt based on this data:{cleaned_text} {prompt_content}
                    """
                }
            ]

            # Generar respuesta
            full_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=600,
            ).choices[0].message["content"]
            sentences = full_response.split(". ")
            response = ". ".join(sentences[0:])
            u_string = unity_string
            f_responose = response
            print('Response:', unity_string + ' ' + response)
            long_response(u_string, f_responose)
            save_to_short_term_memory(question, response, SHORT_TERM_MEMORY_FILE)
            save_to_long_term_memory(question, response, LONG_TERM_MEMORY_FILE)
            
            return jsonify({'response': unity_string + ' ' + response})

        return jsonify({'message': SUCCESFULL_RESPONSE})

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()  
        return jsonify({"error": str(e)})

@app.route('/get_short_response', methods=['GET'])
def get_short_response():
    try:
        if os.path.exists(SHORT_RESPONSE_FILE):
            with open(SHORT_RESPONSE_FILE, 'r') as json_file:
                response_data = json.load(json_file)
                return jsonify(response_data)
        else:
            return jsonify({"error": ERROR_RESPONOSE})
    except FileNotFoundError:
        return jsonify({"error": ERROR_RESPONOSE})

@app.route('/get_long_response', methods=['GET'])
def get_long_response():
    try:
        if os.path.exists(LONG_RESPONSE_FILE):
            with open(LONG_RESPONSE_FILE, 'r') as json_file:
                response_data = json.load(json_file)
                return jsonify(response_data)
        else:
            return jsonify({"error": ERROR_RESPONOSE})
    except FileNotFoundError:
        return jsonify({"error": ERROR_RESPONOSE})

if __name__ == '__main__':
    try:
        clean_json_files(LONG_RESPONSE_FILE)
        with open('responses.json', 'r') as json_file:
            response_data = json.load(json_file)
    except FileNotFoundError:
        response_data = {}
    
    app.run(host='0.0.0.0', port=5000)