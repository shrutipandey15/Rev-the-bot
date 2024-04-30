from flask import Flask, render_template, jsonify, request
import processor
import os

app = Flask(__name__)

# Generate a random secret key
secret_key = os.urandom(24).hex()

# Use the generated secret key in your Flask application
app.config['SECRET_KEY'] = secret_key

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        the_question = request.form['question']
        response = processor.chatbot_response(the_question)
    return jsonify({"response": response })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
