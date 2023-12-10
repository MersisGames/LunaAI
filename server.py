import time
import traceback
import uuid
from flask import Flask, render_template, request, jsonify, session
import json
import os
import openai 

#from luna_ai import generate_short_response as generate_short_response_bot
from luna_ai import generate_response_LTM as generate_response_LTM_bot, load_csv, process_file, read_file_content, save_to_long_term_memory, save_to_short_term_memory, search_emb

app = Flask(__name__)
app.secret_key = os.urandom(24) 

@app.before_request
def before_request():
    if 'csv_selected' not in session:
        session['csv_selected'] = None
        
CLEAN_DATA = "response"
LONG_TERM_MEMORY_FILE = "long_term_memory.txt" 
SUCCESFULL_RESPONSE = "Successfully response generated and saved"
ERROR_RESPONOSE = "No response found"
SHORT_RESPONSE_FILE = 'shortResponse.json'
LONG_RESPONSE_FILE = 'longResponse.json'

response_data = {}
paragraphs = " "
csv_file_server = " "
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
            by generating a response to that prompt based on this data:{CLEAN_DATA} {prompt_content}
                """
            }
        ]

        # Generar respuesta
    start_time = time.time() 
    full_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=100,
    ).choices[0].message["content"]
    sentences = full_response.split(". ")
    response = ". ".join(sentences[0:])
        
    return jsonify({'response': response})



@app.route('/save_files', methods=['POST'])
def save_files():
    try:
        global user_personality
        global user_prompt
        global csv_file_server
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
     
        selected_csv = request.form['csvSelect']
        session['csv_selected'] = selected_csv
        csv_file_server = session['csv_selected']
        print('Selected CSV:', csv_file_server)
        
        return render_template('bot.html', csv_options=csv_files, per_files=per_files, prompt_files=prompt_files)

    except Exception as e:
        return f"Error: {str(e)}"

# def short_response(question):
#     short_response = generate_short_response_bot(question)
#     if os.path.exists(SHORT_RESPONSE_FILE):
#         os.remove(SHORT_RESPONSE_FILE)
#     with open(SHORT_RESPONSE_FILE, 'w') as json_file:
#         json.dump({"response": short_response}, json_file)
#     return jsonify({"message": SUCCESFULL_RESPONSE})

def long_response(question, unity_string):
    long_response = generate_response_LTM_bot(question, SHORT_TERM_MEMORY_FILE, unity_string)
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
    try:
        clean_json_files(LONG_RESPONSE_FILE)
        global user_personality
        global user_prompt
        global csv_file_server
        global CLEAN_DATA

        # Recuperar el nombre del CSV de la sesión
        csv_selected = csv_file_server
        print("Selected:", csv_selected)
        if csv_selected is None:
            return jsonify({'error': 'No CSV selected'})

        #csv_file_path = os.path.join('csv_files', f'{csv_selected}')
        #paragraphs = load_csv(csv_file_path)

        personality_content = user_personality
        prompt_content = user_prompt

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
                by generating a response to that prompt based on this data: {CLEAN_DATA} {prompt_content}
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        print('Response:', unity_string + ' ' + response)
        save_to_short_term_memory(question, response, SHORT_TERM_MEMORY_FILE)
        save_to_long_term_memory(question, response, LONG_TERM_MEMORY_FILE)
            
        return jsonify({'response': unity_string + ' ' + response})


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