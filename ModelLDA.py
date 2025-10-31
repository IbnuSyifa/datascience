import streamlit as st
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import warnings

# --- FUNGSI PREPROCESSING TEKS ---
def preprocess_text(text):
    """
    Membersihkan dan memproses teks input.
    - Tokenisasi & Lowercasing
    - Menghapus stopwords
    - Menghapus token yang terlalu pendek
    """
    result = []
    # simple_preprocess: mengubah dokumen menjadi daftar token lowercase
    for token in simple_preprocess(text, deacc=True):
        # Memeriksa apakah token bukan stopword dan panjangnya lebih dari 3 karakter
        if token not in STOPWORDS and len(token) > 3:
            result.append(token)
    return result

# --- PENGATURAN HALAMAN STREAMLIT ---
st.set_page_config(layout="wide", page_title="Analisis Topik LDA", page_icon="🔎")

# --- JUDUL DAN DESKRIPSI APLIKASI ---
st.title("🔎 Aplikasi Analisis Topik dengan LDA")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk melakukan *topic modeling* menggunakan **Latent Dirichlet Allocation (LDA)** pada dataset Anda sendiri. 
Unggah file CSV, tentukan kolom teks dan jumlah topik, lalu jalankan analisis untuk menemukan topik-topik tersembunyi dalam data Anda.
""")

# --- SIDEBAR UNTUK INPUT PENGGUNA ---
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("1. Unggah File CSV Anda", type=["csv"])
    
    # 2. Input Nama Kolom
    column_name = st.text_input("2. Masukkan Nama Kolom Teks", help="Tuliskan nama kolom yang berisi data teks yang akan dianalisis.")
    
    # 3. Input Jumlah Topik
    num_topics = st.number_input("3. Pilih Jumlah Topik", min_value=2, max_value=30, value=5, step=1, help="Pilih berapa banyak topik yang ingin Anda temukan.")
    
    # 4. Tombol untuk memulai proses
    st.markdown("---")
    start_button = st.button("🚀 Jalankan Analisis")

# --- PANEL UTAMA UNTUK MENAMPILKAN HASIL ---
if start_button and uploaded_file is not None and column_name:
    try:
        # Membaca data dari file yang diunggah
        with st.spinner("Membaca dan memuat data..."):
            df = pd.read_csv(uploaded_file)

        # Validasi nama kolom
        if column_name not in df.columns:
            st.error(f"Error: Kolom '{column_name}' tidak ditemukan. Pastikan nama kolom sudah benar.")
        else:
            # Pra-pemrosesan Data
            with st.spinner("Melakukan pra-pemrosesan teks... Ini mungkin memakan waktu beberapa saat."):
                # Menghapus baris dengan nilai kosong di kolom target
                docs = df[column_name].dropna().astype(str)
                processed_docs = docs.map(preprocess_text)

            if processed_docs.empty:
                st.warning("Tidak ada data yang valid untuk diproses setelah pembersihan. Mohon periksa isi kolom Anda.")
            else:
                # Membuat kamus (dictionary) dan korpus (corpus)
                with st.spinner("Membuat kamus dan korpus..."):
                    dictionary = Dictionary(processed_docs)
                    # Filter kata-kata yang terlalu jarang atau terlalu sering muncul
                    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
                    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

                if not corpus:
                    st.error("Korpus kosong setelah pemfilteran. Coba sesuaikan parameter `filter_extremes` atau periksa kembali data Anda.")
                else:
                    # Melatih Model LDA
                    with st.spinner(f"Melatih model LDA dengan {num_topics} topik..."):
                        lda_model = LdaModel(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=num_topics,
                                             random_state=100,
                                             update_every=1,
                                             chunksize=100,
                                             passes=10,
                                             alpha='auto',
                                             per_word_topics=True)
                    
                    st.success("Analisis berhasil diselesaikan!")
                    
                    # Menampilkan Hasil Topik
                    st.header("📊 Hasil Topik dan Kata Kunci")
                    topics = lda_model.print_topics(num_words=10)
                    for i, topic in enumerate(topics):
                        st.markdown(f"**Topik {i+1}:**")
                        st.write(topic[1])

                    # Visualisasi Interaktif
                    st.header("🌐 Visualisasi Interaktif Topik")
                    st.markdown("""
                    Visualisasi ini membantu Anda menginterpretasikan topik:
                    - **Lingkaran di Kiri**: Setiap lingkaran mewakili satu topik. Ukuran lingkaran menunjukkan seberapa umum topik tersebut.
                    - **Bagan Batang di Kanan**: Menampilkan kata-kata kunci yang paling relevan untuk topik yang dipilih.
                    - **Relevansi Metrik (λ)**: Geser slider untuk menyesuaikan urutan kata kunci antara yang paling khas untuk topik (λ=0) dan yang paling sering muncul (λ=1).
                    """)
                    
                    with st.spinner("Mempersiapkan visualisasi..."):
                        # Menonaktifkan peringatan FutureWarning dari pyLDAvis
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        
                        vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
                        html_string = pyLDAvis.prepared_data_to_html(vis_data)
                        st.components.v1.html(html_string, width=1300, height=800, scrolling=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan: {e}")

# Kondisi jika tombol ditekan tanpa input yang lengkap
elif start_button:
    st.warning("⚠️ Mohon unggah file CSV dan masukkan nama kolom terlebih dahulu.")

# Halaman Awal
else:
    st.info("Selamat datang! Silakan atur parameter di sidebar kiri dan klik 'Jalankan Analisis' untuk memulai.")
    st.markdown("""
    ### Langkah-langkah Penggunaan:
    1.  **Unggah File CSV**: Klik tombol 'Browse files' di sidebar.
    2.  **Masukkan Nama Kolom**: Ketik nama kolom yang berisi data teks.
    3.  **Tentukan Jumlah Topik**: Pilih berapa banyak kelompok topik yang ingin Anda identifikasi.
    4.  **Jalankan Analisis**: Tekan tombol untuk memulai proses.
    """)