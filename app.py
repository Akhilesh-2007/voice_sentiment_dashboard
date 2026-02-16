from flask import Flask, render_template, request, jsonify
import os

import librosa
import speech_recognition as sr
from transformers import pipeline

print("Starting Flask app...")  # DEBUG

app = Flask(__name__)

sentiment_model = None


def get_model():
    """
    Lazily loads and caches the Hugging Face sentiment model.
    """
    global sentiment_model
    if sentiment_model is None:
        print("Loading sentiment model...")
        sentiment_model = pipeline("sentiment-analysis")
        print("Model loaded")
    return sentiment_model


def analyze_audio(audio_path, interval: float = 5.0):
    """
    Analyze an audio file and return an emotion timeline.

    The audio is split into `interval`-second segments. Each segment is:
      1. Transcribed with Google Speech Recognition.
      2. Sent through the sentiment model.
    We then build a timeline that records the emotion at the exact
    minute:second when it changes.
    """
    # Get duration using librosa (best-effort; fall back if it fails)
    try:
        y, sr_rate = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr_rate)
    except Exception as e:
        print("librosa failed to load audio:", e)
        duration = 0.0

    if duration <= 0:
        # Fallback duration so that we at least process one segment
        duration = interval

    recognizer = sr.Recognizer()
    model = get_model()

    timeline = []
    current_time = 0.0
    last_emotion = None

    # Process the audio in small chunks and track emotion changes over time
    with sr.AudioFile(audio_path) as source:
        while current_time < duration:
            remaining = duration - current_time
            segment_duration = min(interval, remaining) if remaining > 0 else interval

            try:
                audio_segment = recognizer.record(source, duration=segment_duration)
            except Exception as e:
                print(f"Recording segment at {current_time:.2f}s failed:", e)
                break

            try:
                text = recognizer.recognize_google(audio_segment)
            except Exception as e:
                print(f"Speech recognition failed at {current_time:.2f}s:", e)
                text = ""

            if text.strip():
                try:
                    sentiment = model(text)[0]
                    emotion = sentiment.get("label", "NEUTRAL")
                    score = float(sentiment.get("score", 0.0))
                except Exception as e:
                    print(f"Sentiment model failed at {current_time:.2f}s:", e)
                    emotion = "NEUTRAL"
                    score = 0.0
            else:
                emotion = "NEUTRAL"
                score = 0.0

            minutes = int(current_time // 60)
            seconds = int(current_time % 60)

            # Only record a point when emotion changes, to pinpoint transitions
            if emotion != last_emotion:
                timeline.append(
                    {
                        "time": f"{minutes:02d}:{seconds:02d}",
                        "emotion": emotion,
                        "score": score,
                    }
                )
                last_emotion = emotion

            current_time += segment_duration

    # Ensure we always return at least one data point
    if not timeline:
        timeline.append({"time": "00:00", "emotion": "NEUTRAL", "score": 0.0})

    return timeline


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("POST request received")

        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio = request.files["audio"]

        if audio.filename == "":
            return jsonify({"error": "Empty audio filename"}), 400

        path = "temp.wav"
        try:
            audio.save(path)
        except Exception as e:
            print("Saving audio failed:", e)
            return jsonify({"error": "Failed to save audio file"}), 500

        try:
            result = analyze_audio(path)
            print("Analysis done")
            return jsonify(result)
        except Exception as e:
            print("Error during analysis:", e)
            return jsonify({"error": "Failed to analyze audio"}), 500
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print("Failed to delete temp file:", e)

    return render_template("index.html")


if __name__ == "__main__":
    # Standard Flask entrypoint for local development.
    # For deployment on Render/Vercel/etc., use a WSGI server
    # (e.g., gunicorn) and point it at `app:app`.
    app.run(host="0.0.0.0", port=5000, debug=True)
