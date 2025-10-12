import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Inverted Index & TF-IDF",
    layout="wide"
)

# --- Fungsi-fungsi Inti (Tidak ada perubahan) ---

def preprocess_text(text):
    """Fungsi sederhana untuk membersihkan teks."""
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text) # Hapus tanda baca
    tokens = text.split() # Tokenisasi (memecah menjadi kata)
    return tokens

@st.cache_data # Cache agar tidak dihitung ulang untuk input yang sama
def create_inverted_index(corpus):
    """Membuat Inverted Index dari corpus."""
    inverted_index = defaultdict(list)
    for doc_id, doc_text in corpus.items():
        tokens = preprocess_text(doc_text)
        for token in set(tokens): # Gunakan set untuk efisiensi
            inverted_index[token].append(doc_id)
    # Urutkan posting list untuk keterbacaan
    for key in inverted_index:
        inverted_index[key].sort()
    return inverted_index

@st.cache_data
def calculate_tfidf(corpus):
    """Menghitung TF-IDF dari corpus."""
    documents_list = list(corpus.values())
    doc_ids = [f"Dokumen {i}" for i in corpus.keys()]
    
    # Menggunakan TfidfVectorizer dari Scikit-learn
    # Menangani kasus jika vocabulary kosong (misal, input hanya angka atau tanda baca)
    try:
        vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: preprocess_text(x))
        tfidf_matrix = vectorizer.fit_transform(documents_list)
        feature_names = vectorizer.get_feature_names_out()
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=doc_ids, columns=feature_names)
        return df_tfidf
    except ValueError:
        # Mengembalikan DataFrame kosong jika tidak ada fitur/kata yang ditemukan
        return pd.DataFrame()


# --- Antarmuka Pengguna (UI) Streamlit ---

st.title("✍️ Interactive Inverted Index & TF-IDF")
st.write("Masukkan beberapa dokumen (satu per baris) untuk membangun **Inverted Index** dan menghitung skor **TF-IDF** secara otomatis.")

# --- Bagian Input Corpus dari Pengguna ---
st.header("1. Masukkan Corpus Anda")
user_input = st.text_area(
    "Tempel atau ketik teks Anda di sini. Pisahkan setiap dokumen dengan baris baru (Enter).",
    height=200,
    placeholder="Dokumen pertama ada di baris ini.\nDokumen kedua di baris selanjutnya.\nContoh: The quick brown fox jumps over the lazy dog."
)

# Tombol untuk memicu analisis
if st.button("Proses dan Analisis Teks"):
    if user_input:
        # Memecah input menjadi dokumen berdasarkan baris baru
        lines = user_input.strip().split('\n')
        
        # Membuat corpus dictionary dari input (hanya baris yang tidak kosong)
        corpus = {i+1: line for i, line in enumerate(lines) if line.strip()}
        
        if not corpus:
            st.warning("Input tidak valid. Pastikan ada teks di setiap baris.")
        else:
            st.success(f"Berhasil memuat {len(corpus)} dokumen. Hasil analisis di bawah ini.")
            
            # --- Tampilkan Corpus yang Dimasukkan ---
            st.header("Corpus yang Anda Masukkan")
            corpus_df = pd.DataFrame(list(corpus.items()), columns=['ID Dokumen', 'Teks'])
            st.table(corpus_df.set_index('ID Dokumen'))
            
            # --- Bagian Inverted Index ---
            st.header("2. Inverted Index")
            st.write("Memetakan setiap kata unik ke daftar ID dokumen di mana kata tersebut muncul.")
            
            inverted_index = create_inverted_index(corpus)
            ii_df = pd.DataFrame(list(inverted_index.items()), columns=['Term', 'Dokumen ID'])
            st.dataframe(ii_df, use_container_width=True)
            
            # --- Bagian TF-IDF ---
            st.header("3. Matriks TF-IDF")
            st.write("Menunjukkan skor pentingnya sebuah kata dalam sebuah dokumen relatif terhadap keseluruhan corpus.")
            
            tfidf_df = calculate_tfidf(corpus)
            if not tfidf_df.empty:
                st.dataframe(tfidf_df.style.format("{:.4f}").background_gradient(cmap='viridis', axis=1), use_container_width=True)
                st.caption("Warna yang lebih terang menunjukkan skor TF-IDF yang lebih tinggi.")
            else:
                st.warning("Tidak dapat menghitung TF-IDF. Pastikan teks Anda mengandung kata-kata yang valid (bukan hanya angka atau simbol).")
    else:
        st.warning("Mohon masukkan teks ke dalam area di atas terlebih dahulu.")
        

st.write("Ibnu Syifa - 241012000087")
