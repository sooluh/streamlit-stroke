import pickle
import streamlit as st

model = pickle.load(open('./datasets/model.sav', 'rb'))

st.set_page_config(
    page_title='Prediksi Risiko Stroke - NeuroInsight Sentinel',
    page_icon='🧮',
)

'''
# 🧠 NeuroInsight Sentinel

**Peringatan!**
Hasil dari prediksi ini belum sempurna dan belum bisa dijadikan rujukan.

---
'''

col1, col2 = st.columns(2)

with col1:
    gender = st.radio('Pilih Gender:', ['Laki-laki', 'Perempuan'])
    gender_binary = 1 if gender == 'Perempuan' else 0

with col2:
    age = st.number_input('Umur:', min_value=0, max_value=85, step=1, value=25)

hypertension = st.checkbox('Apakah Anda memiliki hipertensi?')
hypertension_binary = 1 if hypertension else 0

heart = st.checkbox('Apakah Anda memiliki penyakit jantung?')
heart_binary = 1 if heart else 0

married = st.radio('Status Perkawinan:', [
    'Belum Menikah', 'Sudah Menikah'
])
married_binary = 1 if married == 'Sudah Menikah' else 0

work = st.selectbox('Pilih Tipe Pekerjaan:', [
    'Pegawai Negeri', 'Swasta', 'Wiraswasta', 'Anak-Anak'
])
work_gov = 1 if work == 'Pegawai Negeri' else 0
work_private = 1 if work == 'Swasta' else 0
work_self = 1 if work == 'Wiraswasta' else 0
work_children = 1 if work == 'Anak-Anaka' else 0

residence = st.selectbox('Pilih Tipe Tempat Tinggal:', [
    'Perdesaan', 'Perkotaan'
])
residence_rural = 1 if residence == 'Perdesaan' else 0
residence_urban = 1 if residence == 'Perkotaan' else 0

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input(
        'Rata-rata Gula Darah:', min_value=50, max_value=280, step=1, value=105)

with col2:
    bmi = st.number_input(
        'BMI (Body Mass Index):', min_value=10, max_value=50, step=1, value=28)

smoking = st.selectbox('Seberapa Sering Merokok:', [
    'Tidak Pernah', 'Pernah Merokok', 'Merokok', 'Tidak Tahu'
])
smoking_never = 1 if smoking == 'Tidak Pernah' else 0
smoking_fomerly = 1 if smoking == 'Pernah Merokok' else 0
smoking_smokes = 1 if smoking == 'Merokok' else 0
smoking_unknown = 1 if smoking == 'Tidak Tahu' else 0

submit = st.button('Cek Risiko Kanker')

if submit:
    '''
    ---
    '''

    data = [
        gender_binary, age, hypertension_binary, heart_binary, married_binary,
        glucose, bmi, work_gov, work_private, work_self, work_children,
        residence_rural, residence_urban, smoking_unknown, smoking_fomerly,
        smoking_never, smoking_smokes,
    ]

    result = model.predict([data])

    if result[0]:
        st.warning("Innalillahi, kami mendeteksi risiko kanker.")
    else:
        st.success("Alhamdulillah, kami tidak mendeteksi risiko kanker.")
