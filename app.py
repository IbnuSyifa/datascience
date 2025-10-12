from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)

# Dokumen sampel (kumpulan kalimat)
documents = [
    "Machine learning is a subfield of artificial intelligence",
    "Deep learning is a subfield of machine learning",
    "Artificial intelligence and machine learning are popular fields",
    "Natural language processing is a part of artificial intelligence",
    "TF-IDF is a common technique in natural language processing"
]

# Inisialisasi TfidfVectorizer
# TfidfVectorizer akan menangani tokenisasi, menghitung TF, dan IDF
vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True)

# Menghitung matriks TF-IDF dari dokumen
tfidf_matrix = vectorizer.fit_transform(documents)

# Mengambil nama fitur (kata-kata unik)
feature_names = vectorizer.get_feature_names_out()

# Mengubah matriks TF-IDF menjadi DataFrame pandas agar lebih mudah dibaca
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=[f"Dokumen {i+1}" for i in range(len(documents))])


@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    results = None
    error = None

    if request.method == 'POST':
        query = request.form.get('query', '').lower()
        if query:
            if query in df_tfidf.columns:
                # Jika kata ditemukan, ambil skornya
                results = df_tfidf[query].to_dict()
            else:
                # Jika kata tidak ditemukan di vocabulary
                error = f"Kata '{query}' tidak ditemukan dalam dokumen."
        else:
            error = "Mohon masukkan kata untuk dicari."

    # Mengirimkan data ke template HTML
    # 'tables' mengirimkan DataFrame TF-IDF dalam format HTML
    # 'columns' mengirimkan header tabel
    return render_template(
        'index.html',
        tables=[df_tfidf.to_html(classes='table table-hover', header="true")],
        columns=df_tfidf.columns,
        query=query,
        results=results,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)