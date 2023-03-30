import os
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_transcript(audio_file):
    response = openai.Audio.create(
        file=audio_file.stream,
        purpose="transcription",
    )

    transcript = response.get("text")
    return transcript

def get_summary(transcript):
    prompt = f"Please provide a summary of the following transcript:\n{transcript}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    summary = response.choices[0].text.strip()
    return summary

@app.route('/api/summarize', methods=['POST'])
def summarize_audio():
    audio_file = request.files['audio']
    transcript = get_transcript(audio_file)
    summary = get_summary(transcript)
    return jsonify(summary=summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
