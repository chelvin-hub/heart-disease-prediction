import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_clean.csv")

model = load_model()
df = load_data()

# Sidebar
st.sidebar.title("🫀 Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Home",
        "Dataset",
        "EDA",
        "Prediksi",
        "Tentang Model"
    ]
)

# =========================
# HOME
# =========================

if menu == "Home":

    st.title("🫀 Heart Disease Prediction")

    st.write("""
    Aplikasi ini digunakan untuk melakukan prediksi penyakit jantung
    menggunakan Machine Learning dengan algoritma CatBoost Classifier.
    """)

    st.subheader("Tujuan Aplikasi")

    st.write("""
    Aplikasi ini dibuat untuk membantu melakukan prediksi berdasarkan
    data kesehatan pasien.
    """)

    st.subheader("Algoritma")

    st.write("CatBoost Classifier")


# =========================
# DATASET
# =========================

elif menu == "Dataset":

    st.title("📊 Dataset")

    st.write("Dataset yang digunakan dalam aplikasi.")

    st.dataframe(df)

    st.subheader("Informasi Dataset")

    st.write("Jumlah Baris:", df.shape[0])

    st.write("Jumlah Kolom:", df.shape[1])


# =========================
# EDA
# =========================

elif menu == "EDA":

    st.title("📈 Exploratory Data Analysis")

    st.subheader("Statistik Dataset")

    st.dataframe(df.describe())

    st.subheader("Distribusi Target")

    if "target" in df.columns:

        fig, ax = plt.subplots()

        df["target"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)


# =========================
# PREDIKSI
# =========================

elif menu == "Prediksi":

    st.title("🤖 Prediksi Penyakit Jantung")

    st.write("Masukkan data pasien di bawah ini.")

    age = st.number_input(
        "Umur",
        min_value=1,
        max_value=100,
        value=50
    )

   sex_option = st.selectbox(
    "Jenis Kelamin",
    ["Perempuan", "Laki-laki"]
)

sex = 0 if sex_option == "Perempuan" else 1

  cp_option = st.selectbox(
    "Jenis Nyeri Dada",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_mapping[cp_option]

    trestbps = st.number_input(
        "Tekanan Darah",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Kolesterol",
        min_value=100,
        max_value=600,
        value=200
    )

  fbs_option = st.selectbox(
    "Gula Darah Puasa",
    [
        "Normal",
        "Tinggi (> 120 mg/dl)"
    ]
)

fbs = 0 if fbs_option == "Normal" else 1

  restecg_option = st.selectbox(
    "Hasil ECG Saat Istirahat",
    [
        "Normal",
        "Kelainan ST-T",
        "Hipertrofi Ventrikel Kiri"
    ]
)

restecg_mapping = {
    "Normal": 0,
    "Kelainan ST-T": 1,
    "Hipertrofi Ventrikel Kiri": 2
}

restecg = restecg_mapping[restecg_option]

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

  exang_option = st.selectbox(
    "Nyeri Dada Saat Berolahraga",
    [
        "Tidak",
        "Ya"
    ]
)

exang = 0 if exang_option == "Tidak" else 1

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

  slope_option = st.selectbox(
    "Slope ST",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope = slope_mapping[slope_option]

   ca_option = st.selectbox(
    "Jumlah Pembuluh Darah Utama",
    [
        "0 Pembuluh",
        "1 Pembuluh",
        "2 Pembuluh",
        "3 Pembuluh",
        "4 Pembuluh"
    ]
)

ca_mapping = {
    "0 Pembuluh": 0,
    "1 Pembuluh": 1,
    "2 Pembuluh": 2,
    "3 Pembuluh": 3,
    "4 Pembuluh": 4
}

ca = ca_mapping[ca_option]

   thal_option = st.selectbox(
    "Hasil Thal",
    [
        "Unknown",
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

thal_mapping = {
    "Unknown": 0,
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_mapping[thal_option]

    # Membuat data input
    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    if st.button("Prediksi"):

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.success("Hasil Prediksi: Memiliki indikasi penyakit jantung")

        else:

            st.info("Hasil Prediksi: Tidak memiliki indikasi penyakit jantung")


# =========================
# TENTANG MODEL
# =========================

elif menu == "Tentang Model":

    st.title("ℹ️ Tentang Model")

    st.write("""
    Model Machine Learning yang digunakan adalah CatBoost Classifier.

    Dataset dibagi menjadi:

    - 80% Data Training
    - 20% Data Testing

    Evaluasi model dilakukan menggunakan:

    - Accuracy
    - Classification Report
    - Confusion Matrix
    - ROC Curve
    - AUC Score
    """)
