from flask import Flask, render_template, request
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vect)[0]
    return prediction


if __name__ == "__main__":
    app.run(debug=True)