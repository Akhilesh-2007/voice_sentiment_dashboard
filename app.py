from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from transformers import pipeline
import librosa

app = Flask(__name__)
sentiment_model = pipeline("sentiment-analysis")

def analyze_audio(audio_path):
    y, sr_rate = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr_rate)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

    sentiment = sentiment_model(text)

    timeline = []
    interval = 5
    current_time = 0
    for _ in range(int(duration // interval)):
        timeline.append({
            "time": f"{current_time//60}:{current_time%60}",
            "emotion": sentiment[0]["label"]
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
    app.run(debug=True)
