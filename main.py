import speech_recognition as sr
from resume_data import resume_data
from fuzzywuzzy import fuzz
import pyttsx3

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening... Ask a question.")
        audio_data = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio_data)
            print("🗣️ You said:", text)
            return text.lower()
        except sr.UnknownValueError:
            return "Sorry, I didn't understand that."
        except sr.RequestError:
            return "Network error. Please check your internet connection."

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight transformer

def get_response(query):
    keys = list(resume_data.keys())
    key_embeddings = model.encode(keys, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Find the most similar key
    scores = util.pytorch_cos_sim(query_embedding, key_embeddings)[0]
    best_match_index = scores.argmax().item()
    best_score = scores[best_match_index].item()

    best_key = keys[best_match_index]
    if best_score > 0.4:  # threshold to prevent wrong matches
        value = resume_data[best_key]
        if isinstance(value, list):
            value = ", ".join(value)
        return f"My {best_key} is {value}"
    else:
        return "Sorry, I don't have information about that yet."


def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        query = listen_to_user()
        if query in ["exit", "quit", "bye"]:
            speak_text("Goodbye! Have a nice day.")
            print("👋 Exiting the chatbot.")
            break
        response = get_response(query)
        print("🤖 Bot:", response)
        speak_text(response)
