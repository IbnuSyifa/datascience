import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- KONFIGURASI HALAMAN ---
# Menyetel konfigurasi halaman harus menjadi perintah pertama Streamlit
st.set_page_config(
    page_title="TF-IDF Search Engine",
    page_icon="🔎",
    layout="centered" # Pilihan lain: "wide"
)

# --- FUNGSI UTAMA & PERHITUNGAN TF-IDF ---
# Logika ini sama persis dengan versi Flask
documents = [
    "Pembelajaran mesin adalah subbidang dari kecerdasan buatan.",
    "Deep learning adalah subbidang dari machine learning.",
    "Kecerdasan buatan dan pembelajaran mesin adalah bidang yang populer.",
    "Pemrosesan bahasa alami merupakan bagian dari kecerdasan buatan.",
    "TF-IDF adalah teknik yang umum digunakan dalam pemrosesan bahasa alami."
]

@st.cache_data # Menambahkan cache agar tidak perlu menghitung ulang setiap kali ada interaksi
def calculate_tfidf():
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=[f"Dokumen {i+1}" for i in range(len(documents))]
    )
    return df_tfidf

df_tfidf = calculate_tfidf()

# --- ANTARMUKA PENGGUNA (UI) STREAMLIT ---

st.title("🔎 TF-IDF Calculator & Search")
st.write("Aplikasi web sederhana untuk menghitung dan mencari skor TF-IDF dari sekumpulan dokumen.")


# Menampilkan Dokumen Sampel
with st.expander("Lihat Dokumen Sampel"):
    for i, doc in enumerate(documents):
        st.write(f"**Dokumen {i+1}:** *{doc}*")

st.header("Pencarian Skor TF-IDF")

# Input dari pengguna
query = st.text_input(
    "Masukkan satu kata untuk dicari:",
    placeholder="Contoh: learning, intelligence, a"
).lower()

# Tombol untuk memicu pencarian
if st.button("Cari"):
    if query:
        if query in df_tfidf.columns:
            st.success(f"Hasil pencarian untuk kata: **'{query}'**")
            # Mengambil hasil skor dan mengubahnya menjadi DataFrame untuk tampilan yang lebih baik
            results_df = df_tfidf[[query]].sort_values(by=query, ascending=False)
            st.dataframe(results_df)
        else:
            st.error(f"Kata '{query}' tidak ditemukan dalam vocabulary dokumen.")
    else:
        st.warning("Mohon masukkan sebuah kata untuk dicari.")


# Menampilkan Matriks TF-IDF Lengkap
st.header("Matriks TF-IDF Lengkap")
st.dataframe(df_tfidf)

st.caption("Tabel ini menunjukkan skor TF-IDF untuk setiap kata (kolom) di setiap dokumen (baris).")

st.write("Ibnu Syifa - 241012000087")
