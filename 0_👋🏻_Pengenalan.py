import streamlit as st

st.set_page_config(
    page_title="Pengenalan - NeuroInsight Sentinel",
    page_icon="👋🏻",
)

'''
# 🧠 NeuroInsight Sentinel

Platform ini menyajikan proyek terkini di bidang kesehatan dan pencegahan penyakit, dengan fokus
pada identifikasi risiko stroke pada individu. Menggunakan teknologi machine learning, proyek ini
bertujuan mengembangkan model prediktif yang akurat dengan mempertimbangkan faktor-faktor kritis
seperti jenis kelamin, umur, kondisi kesehatan, dan gaya hidup. Proyek kami bermaksud memberikan
kontribusi dalam meningkatkan deteksi dini risiko stroke, sehingga memungkinkan intervensi yang
lebih efektif.

Masuk ke menu Prediksi untuk memulai.

**Who Am I?**
'''

st.markdown(
    '<table width="100%" style="margin-bottom: 1rem;"><tr><td style="width: 1%; white-space: nowrap;">Nama</td><td style="width: 1%; white-space: nowrap;">:</td><td>Suluh Sulistiawan</td></tr><tr><td>NIM</td><td>:</td><td>211351143</td></tr><tr><td>Kelas</td><td>:</td><td>Informatika Malam B</td></tr></table>',
    unsafe_allow_html=True,
)

'''
Di bawah adalah hasil confusion matrix untuk model yang digunakan pada platform ini.
'''

st.image('./assets/confusion_matrix.png')

'''
Ini adalah proyek sumber terbuka: [@sooluh/streamlit-stroke](https://github.com/sooluh/streamlit-stroke)
'''
