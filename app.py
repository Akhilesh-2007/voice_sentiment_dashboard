from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from transformers import pipeline
import librosa
import os

print("Starting Flask app...")  # DEBUG

app = Flask(__name__)

sentiment_model = None

def get_model():
    global sentiment_model
    if sentiment_model is None:
        print("Loading sentiment model...")
        sentiment_model = pipeline("sentiment-analysis")
        print("Model loaded")
    return sentiment_model


def analyze_audio(audio_path):
    y, sr_rate = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr_rate)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except Exception as e:
            print("Speech recognition failed:", e)
            text = "Neutral speech"

    model = get_model()
    sentiment = model(text)

    timeline = []
    interval = 5
    current_time = 0

    for _ in range(max(1, int(duration // interval))):
        timeline.append({
            "time": f"{current_time//60}:{current_time%60:02d}",
            "emotion": sentiment[0]["label"]
        })
        current_time += interval

    return timeline


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("POST request received")

        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio = request.files["audio"]
        path = "temp.wav"
        audio.save(path)

        result = analyze_audio(path)
        print("Analysis done")

        return jsonify(result)

    return render_template("index.html")


# 🔴 IMPORTANT: DO NOT USE __name__ CHECK
app.run(host="0.0.0.0", port=5000, debug=True)
