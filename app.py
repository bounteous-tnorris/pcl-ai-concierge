
import json
import openai
from collections import deque
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

messages = [{"role": "system", "content": "You are a helpful assistant."}]

def answer(msg):
    global messages
    prompt = msg
    messages = messages + [{"role": "user", "content": prompt}]
    
    model = "gpt-4-turbo-preview"
    completions = openai.chat.completions.create(model=model, messages=messages, max_tokens=1024, n=1,stop=None, temperature=0.7)
    response = completions.choices[0].message.content
    messages = messages + [{"role": "assistant", "content": response}]
    [print(m) for m in messages]
    return response


from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return answer(userText)

if __name__ == "__main__":
    app.run()
