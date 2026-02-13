from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from textblob import TextBlob
import librosa
import os

app = Flask(__name__)

def analyze_audio(audio_path):
    y, sr_rate = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr_rate)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

    timeline = []
    interval = 5
    current_time = 0

    for _ in range(int(duration // interval)):
        polarity = TextBlob(text).sentiment.polarity
        emotion = "POSITIVE" if polarity > 0 else "NEGATIVE"

        timeline.append({
            "time": f"{current_time//60}:{current_time%60}",
            "emotion": emotion
        })
        current_time += interval

    return timeline

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio = request.files["audio"]
        path = "temp.wav"
        audio.save(path)
        result = analyze_audio(path)
        return jsonify(result)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
