import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk memuat dataset
@st.cache
def load_data():
    kaggle_df = pd.read_csv('data/kaggle/drug_consumption.csv')
    uci_df = pd.read_csv('data/uci/drug_consumption.data', header=None)
    
    # Menyusun kolom untuk UCI dataset
    uci_columns = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 
                   'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS'] + [f'Drug_{i}' for i in range(1, 20)]
    uci_df.columns = uci_columns
    
    return kaggle_df, uci_df

# Preprocessing data
def preprocess_data(kaggle_df, uci_df):
    # Drop columns that are not common between datasets
    common_columns = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 
                      'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    kaggle_df = kaggle_df[common_columns + list(kaggle_df.columns[13:])]
    uci_df = uci_df[common_columns + list(uci_df.columns[13:])]
    
    # Concatenate datasets
    combined_df = pd.concat([kaggle_df, uci_df], axis=0)
    
    # Encode categorical variables
    for col in combined_df.select_dtypes(include=['object']).columns:
        combined_df[col] = combined_df[col].astype('category').cat.codes
    
    X = combined_df[common_columns]
    y = combined_df.drop(columns=common_columns)
    
    return X, y

# Fungsi untuk melatih model
def train_model(X, y):
    # Memisahkan data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model Random Forest untuk setiap jenis narkoba
    models = {}
    accuracies = {}
    reports = {}
    for col in y.columns:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train[col])
        y_pred = rf.predict(X_test)
        models[col] = rf
        accuracies[col] = accuracy_score(y_test[col], y_pred)
        reports[col] = classification_report(y_test[col], y_pred, output_dict=True)
    
    return models, accuracies, reports

# Fungsi untuk menampilkan hasil
# def display_results(accuracies, reports):
#     st.write("### Akurasi Model")
#     for drug, accuracy in accuracies.items():
#         st.write(f"{drug}: {accuracy:.2f}")

#     st.write("### Laporan Klasifikasi")
#     for drug, report in reports.items():
#         st.write(f"#### {drug}")
#         st.json(report)

# Fungsi untuk menampilkan hasil
def display_results(accuracies, reports, models, X):
    st.write("### Akurasi Model")
    for drug, accuracy in accuracies.items():
        st.write(f"{drug}: {accuracy:.2f}")

    st.write("### Laporan Klasifikasi")
    for drug, report in reports.items():
        st.write(f"#### {drug}")
        st.json(report)

    st.write("### Pentingnya Fitur untuk Setiap Jenis Narkoba")
    for drug, model in models.items():
        st.write(f"#### {drug}")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plot pentingnya fitur
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {drug}')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()


# Streamlit UI
st.title("Analisis Penggunaan Narkoba dengan Random Forest")
st.write("Proyek ini menggunakan algoritma Random Forest untuk menganalisis pola penggunaan narkoba.")

# Memuat data
data_load_state = st.text('Memuat data...')
kaggle_data, uci_data = load_data()
data_load_state.text('Data dimuat!')

# Menampilkan beberapa baris data
if st.checkbox('Tampilkan data mentah'):
    st.subheader('Data Kaggle')
    st.write(kaggle_data.head())
    st.subheader('Data UCI')
    st.write(uci_data.head())

# Preprocessing data
X, y = preprocess_data(kaggle_data, uci_data)

# Melatih model dan menampilkan hasil
if st.button('Latih Model'):
    st.text('Melatih model...')
    models, accuracies, reports = train_model(X, y)
    st.text('Model selesai dilatih!')
    display_results(accuracies, reports, models, X)
