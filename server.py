import uuid
from flask import Flask, request, jsonify
import json
import os 

from luna_ai import generate_short_response as generate_short_response_bot
from luna_ai import generate_response_LTM as generate_response_LTM_bot

app = Flask(__name__)
SUCCESFULL_RESPONSE = "Successfully response generated and saved"
ERROR_RESPONOSE = "No response found"
SHORT_RESPONSE_FILE = 'shortResponse.json'
LONG_RESPONSE_FILE = 'longResponse.json'

response_data = {}

SHORT_TERM_MEMORY_FILE = str(uuid.uuid4()) + "_STM.txt" 

def short_response(question):
    short_response = generate_short_response_bot(question)
    if os.path.exists(SHORT_RESPONSE_FILE):
        os.remove(SHORT_RESPONSE_FILE)
    with open(SHORT_RESPONSE_FILE, 'w') as json_file:
        json.dump({"response": short_response}, json_file)
    return jsonify({"message": SUCCESFULL_RESPONSE})

def long_response(question):
    long_response = generate_response_LTM_bot(question, SHORT_TERM_MEMORY_FILE)
    if os.path.exists(LONG_RESPONSE_FILE):
        os.remove(LONG_RESPONSE_FILE)
    with open(LONG_RESPONSE_FILE, 'w') as json_file:
        json.dump({"response": long_response}, json_file)
    return jsonify({"message": SUCCESFULL_RESPONSE })


@app.route('/ask_question/<question>', methods=['POST'])
def ask_question(question):
    try:
        short_response(question)
        long_response(question)
        return jsonify({"message": SUCCESFULL_RESPONSE })
    except Exception as e:
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
        with open('responses.json', 'r') as json_file:
            response_data = json.load(json_file)
    except FileNotFoundError:
        response_data = {}
    
    app.run(host='0.0.0.0', port=5000)