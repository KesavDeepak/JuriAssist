from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from google import genai


nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)
USER_FILE = 'users.json'
app.secret_key = 'kesav_123'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'static', 'ipc_sections.csv')

df = pd.read_csv(csv_path)
df["Offense"] = df["Offense"].fillna("None").str.strip().str.lower()
df["Description"] = df["Description"].fillna("").str.strip().str.lower()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["Processed"] = (df["Offense"] + " " + df["Description"]).apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Processed"])
tfidf_vocab = list(vectorizer.get_feature_names_out())

def correct_query(query):
    words = query.split()
    correction_vocab = tfidf_vocab + ["divorce"]
    corrected = []
    for w in words:
        match, score, _ = process.extractOne(w, correction_vocab, scorer=fuzz.WRatio)
        corrected.append(match if score > 85 else w)
    return " ".join(corrected)

def classical_scores(query):
    query = preprocess_text(correct_query(query))
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    return scores

def quantum_grover_search(scores, top_k=5):
    probs = scores / np.sum(scores)
    n = max(1, int(np.ceil(np.log2(len(probs)))))
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    backend = Aer.get_backend("qasm_simulator")
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=2000)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return df.iloc[top_indices][["Section", "Offense", "Description"]]


if os.path.exists('trained_model.pkl'):
    with open('trained_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    sentence_model = model_data['model']
    kmeans = model_data['kmeans']
    cluster_labels = model_data['cluster_labels']
    judgements_df = model_data['data']
    embeddings = model_data['embeddings']
else:
    sentence_model = None
    kmeans = None
    cluster_labels = {}
    judgements_df = pd.DataFrame()
    embeddings = None
    print("⚠️ trained_model.pkl not found. Run train_model.py first.")


@app.route("/search", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        query = request.form.get("query")
        scores = classical_scores(query)
        results = quantum_grover_search(scores, top_k=5).to_dict(orient="records")
    return render_template("search.html", results=results)

@app.route('/home_page')
def home_page():
    return render_template('home.html')

@app.route('/')
def search():
    return render_template('index.html')
with open('trained_model.pkl', 'rb') as f:
    trained_data = pickle.load(f)

mmodel = trained_data['model']
kmeans = trained_data['kmeans']
cluster_labels = trained_data['cluster_labels']
judgements_df = trained_data['data']
embeddings = trained_data['embeddings']

@app.route('/judgements', methods=['GET', 'POST'])
def judgements():
    results = []
    cluster_label = None

    if request.method == 'POST':
        query = request.form['query']

        query_embedding = mmodel.encode([query])
        query_cluster = kmeans.predict(query_embedding)[0]
        cluster_label = cluster_labels.get(query_cluster, "Unknown Topic")

        cluster_cases = judgements_df[judgements_df['cluster'] == query_cluster]
        cluster_embeddings = embeddings[judgements_df['cluster'] == query_cluster]
        similarities = np.dot(cluster_embeddings, query_embedding.T).flatten()
        top_indices = similarities.argsort()[-5:][::-1]

        for idx in top_indices:
            row = cluster_cases.iloc[idx]
            results.append({
                'case_name': row['case_name'],
                'judgement_date': row.get('judgement_date', 'Unknown'),
                'answer': row['answer']
            })

    return render_template('judgement.html', results=results, cluster_label=cluster_label)


@app.route('/bookmark')
def bookmark():
    return render_template('bookmark.html')

AI_KEY = "AIzaSyClW-XMm5OLLpWrXVjQWLhaEJuAGTr5Ic4"


client = genai.Client(api_key=AI_KEY)

@app.route('/clueai', methods=['GET', 'POST'])
def clueai():
    ai_response = None
    if request.method == 'POST':
        query = request.form.get('query')

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": query}]}],
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024
                )
            )
            if response.text:
                ai_response = response.text
            else:
                if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                    ai_response = "The query or generated response was blocked due to safety settings. Please try a different question."
                else:
                    finish_reason = response.candidates[0].finish_reason.name if response.candidates and response.candidates[0].finish_reason else "N/A"
                    ai_response = f"The model returned an empty response. Finish Reason: {finish_reason}"
 
            
        except Exception as e:
            ai_response = f"Error calling Gemini API: {e}"

    return render_template('clueai.html', response=ai_response)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part found!')
        return redirect(url_for('bookmark'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!')
        return redirect(url_for('bookmark'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    flash(f'✅ File "{file.filename}" uploaded successfully!')
    return redirect(url_for('bookmark'))
@app.route('/save_ipc', methods=['POST'])
def save_ipc():
    ipc_text = request.form.get("ipc_text", "").strip()

    if ipc_text:
        with open("ipc_sections.txt", "a", encoding="utf-8") as f:
            f.write(ipc_text + "\n\n")

        flash("✅ IPC sections saved successfully!")

    return redirect(url_for("bookmark"))


@app.route('/save_judgement', methods=['POST'])
def save_judgement():
    judgement_text = request.form.get("judgement_text", "").strip()

    if judgement_text:
        with open("judgements.txt", "a", encoding="utf-8") as f:
            f.write(judgement_text + "\n\n")

        flash("✅ Judgement saved successfully!")

    return redirect(url_for("bookmark"))

@app.route('/view_files')
def view_files():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])

    ipc_sections = []
    if os.path.exists("ipc_sections.txt"):
        with open("ipc_sections.txt", "r", encoding="utf-8") as f:
            ipc_sections = [line.strip() for line in f if line.strip()]

    judgements = []
    if os.path.exists("judgements.txt"):
        with open("judgements.txt", "r", encoding="utf-8") as f:
            judgements = [line.strip() for line in f if line.strip()]

    return render_template(
        "view_files.html",
        uploaded_files=uploaded_files,
        ipc_sections=ipc_sections,
        judgements=judgements
    )

from flask import send_from_directory

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username] == password:
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('home_page'))
        else:
            flash("Invalid username or password!")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            flash("Username already exists!")
            return redirect(url_for('signup'))
        users[username] = password
        save_users(users)
        flash("Signup successful! Please login.")
        return redirect(url_for('login'))
    return render_template('signup.html')

if __name__ == "__main__":
    app.run(debug=True)
