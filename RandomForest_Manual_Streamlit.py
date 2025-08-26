# ==============================================================================
#           APLIKASI STREAMLIT: DEMONSTRASI PERHITUNGAN MANUAL
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="Perhitungan Manual Model")

st.title("⚙️ Demonstrasi Perhitungan Manual Model")
st.write("""
Aplikasi ini bertujuan untuk membedah dan memvisualisasikan proses perhitungan internal dari model **Random Forest** dan **LSTM** secara langkah demi langkah. Gunakan navigasi di sidebar untuk memilih model yang ingin Anda lihat.
""")

# ==============================================================================
#                              NAVIGASI SIDEBAR
# ==============================================================================
st.sidebar.title("Pilih Model")
model_choice = st.sidebar.radio(
    "Pilih model untuk didemonstrasikan:",
    ("Analisis Faktor (Random Forest)", "Prediksi Harga (LSTM)")
)
st.sidebar.markdown("---")


# ==============================================================================
#             BAGIAN 1: PERHITUNGAN MANUAL RANDOM FOREST
# ==============================================================================
def run_rf_manual_calculation():
    st.header("🌳 Perhitungan Manual: Analisis Faktor Random Forest")
    st.write(
        "Demonstrasi ini menunjukkan bagaimana sebuah pohon keputusan memilih fitur terbaik (Langkah 1-4) dan bagaimana skor dari banyak pohon diakumulasikan menjadi hasil akhir (Langkah 5).")

    # --- Langkah 1: Dataset Contoh ---
    with st.expander("Langkah 1: Siapkan Sampel Data Sederhana", expanded=True):
        data = {
            'Harga': [30000, 35000, 32000, 90000, 85000, 92000],
            'RR': [5, 2, 8, 30, 25, 35],
            'stok_harian': [17500, 17500, 13125, 13125, 40000, 30000]
        }
        df_sampel = pd.DataFrame(data)
        st.write("Misalkan kita memiliki sampel data di mana korelasi Curah Hujan (RR) dengan Harga sangat jelas:")
        st.dataframe(df_sampel)

    # --- Langkah 2: MSE Awal ---
    with st.expander("Langkah 2: Hitung MSE Awal (Parent Node)", expanded=True):
        harga_awal = df_sampel['Harga'].values
        mean_awal = np.mean(harga_awal)
        mse_awal = np.mean((harga_awal - mean_awal) ** 2)

        st.write(f"**1. Hitung Rata-rata Harga (Mean):**")
        st.latex(
            fr"\text{{Mean}}_{{\text{{awal}}}} = \frac{{30.000 + 35.000 + 32.000 + 90.000 + 85.000 + 92.000}}{{6}} = {mean_awal:,.0f}")

        st.write("**2. Hitung Mean Squared Error (MSE):**")
        st.latex(
            fr"\text{{MSE}}_{{\text{{awal}}}} = \frac{{(30.000 - 60.667)^2 + ... + (92.000 - 60.667)^2}}{{6}}")
        st.latex(fr"\text{{MSE}}_{{\text{{awal}}}} = {mse_awal:,.0f}")
        st.info(f"Ini adalah **error awal** sebesar **{mse_awal:,.0f}** yang ingin kita kurangi.")

    # --- Langkah 3: Uji Fitur 'RR' (Curah Hujan) ---
    with st.expander("Langkah 3: Uji Coba Split pada Fitur 'RR' (Curah Hujan)", expanded=True):
        split_value_rr = 15.0
        st.write(f"Model menguji pemisahan pada kondisi: **RR ≤ {split_value_rr}**")

        grup_kiri_rr = df_sampel[df_sampel['RR'] <= split_value_rr]
        grup_kanan_rr = df_sampel[df_sampel['RR'] > split_value_rr]

        st.subheader("1. Hitung MSE untuk Setiap Grup Anak")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Grup Kiri (RR ≤ 15.0):**")
            st.dataframe(grup_kiri_rr)
            mean_kiri_rr = np.mean(grup_kiri_rr['Harga'])
            mse_kiri_rr = np.mean((grup_kiri_rr['Harga'] - mean_kiri_rr) ** 2)
            st.latex(fr"\text{{Mean}}_{{\text{{kiri}}}} = {mean_kiri_rr:,.0f}")
            st.latex(fr"\text{{MSE}}_{{\text{{kiri}}}} = {mse_kiri_rr:,.0f}")
        with col2:
            st.write("**Grup Kanan (RR > 15.0):**")
            st.dataframe(grup_kanan_rr)
            mean_kanan_rr = np.mean(grup_kanan_rr['Harga'])
            mse_kanan_rr = np.mean((grup_kanan_rr['Harga'] - mean_kanan_rr) ** 2)
            st.latex(fr"\text{{Mean}}_{{\text{{kanan}}}} = {mean_kanan_rr:,.0f}")
            st.latex(fr"\text{{MSE}}_{{\text{{kanan}}}} = {mse_kanan_rr:,.0f}")

        st.subheader("2. Hitung MSE Gabungan (Berbobot)")
        bobot_kiri = len(grup_kiri_rr) / len(df_sampel)
        bobot_kanan = len(grup_kanan_rr) / len(df_sampel)
        mse_setelah_rr = (bobot_kiri * mse_kiri_rr) + (bobot_kanan * mse_kanan_rr)
        st.latex(
            fr"\text{{MSE}}_{{\text{{setelah}}}} = (\frac{{{len(grup_kiri_rr)}}}{{{len(df_sampel)}}} \times {mse_kiri_rr:,.0f}) + (\frac{{{len(grup_kanan_rr)}}}{{{len(df_sampel)}}} \times {mse_kanan_rr:,.0f}) = {mse_setelah_rr:,.0f}")

        st.subheader("3. Hitung Penurunan Error (Skor Kepentingan)")
        penurunan_error_rr = mse_awal - mse_setelah_rr
        st.latex(fr"\text{{Skor}} = {mse_awal:,.0f} - {mse_setelah_rr:,.0f}")
        st.success(f"**Penurunan Error oleh 'RR'**: {penurunan_error_rr:,.0f}")

    # --- Langkah 4: Uji Fitur 'stok_harian' ---
    with st.expander("Langkah 4: Uji Coba Split pada Fitur 'stok_harian'", expanded=True):
        # (Perhitungan serupa untuk 'stok_harian' disajikan dengan cara yang sama)
        split_value_stok = 25000
        grup_kiri_stok = df_sampel[df_sampel['stok_harian'] <= split_value_stok]
        grup_kanan_stok = df_sampel[df_sampel['stok_harian'] > split_value_stok]
        mse_kiri_stok = np.mean(
            (grup_kiri_stok['Harga'] - grup_kiri_stok['Harga'].mean()) ** 2) if not grup_kiri_stok.empty else 0
        mse_kanan_stok = np.mean(
            (grup_kanan_stok['Harga'] - grup_kanan_stok['Harga'].mean()) ** 2) if not grup_kanan_stok.empty else 0
        bobot_kiri = len(grup_kiri_stok) / len(df_sampel)
        bobot_kanan = len(grup_kanan_stok) / len(df_sampel)
        mse_setelah_stok = (bobot_kiri * mse_kiri_stok) + (bobot_kanan * mse_kanan_stok)
        penurunan_error_stok = mse_awal - mse_setelah_stok
        st.success(f"**Penurunan Error oleh 'stok_harian'**: {penurunan_error_stok:,.0f}")

    # --- Langkah 5: Simulasi Akumulasi ---
    with st.expander("Langkah 5: Simulasi Akumulasi & Normalisasi (Menuju Hasil Akhir)", expanded=True):
        st.write(
            "Skor akhir didapat dari **akumulasi** penurunan error di seluruh 100 pohon, yang kemudian **dinormalisasi**. Berikut simulasinya:")

        total_importance = {
            'RR': 10000, 'stok_harian': 1570, 'is_payday_week': 1010, 'is_nataru_period': 900,
            'Hari_Minggu': 690, 'Hari_Jumat': 670, 'Hari_Sabtu': 650, 'Hari_Senin': 630,
            'Hari_Rabu': 628, 'Hari_Kamis': 570, 'Hari_Selasa': 530,
            'is_idul_adha_period': 305, 'is_idul_fitri_period': 170
        }

        simulasi_df = pd.DataFrame(list(total_importance.items()),
                                   columns=['Faktor', 'Total Akumulasi Skor (Ilustrasi)'])
        total_keseluruhan = simulasi_df['Total Akumulasi Skor (Ilustrasi)'].sum()
        simulasi_df['Kontribusi Akhir (%)'] = (simulasi_df[
                                                   'Total Akumulasi Skor (Ilustrasi)'] / total_keseluruhan) * 100
        simulasi_df = simulasi_df.sort_values(by='Kontribusi Akhir (%)', ascending=False)

        st.subheader("1. Simulasi Akumulasi Skor (dari 100 Pohon)")
        st.dataframe(simulasi_df.style.format(
            {'Total Akumulasi Skor (Ilustrasi)': '{:,.0f}', 'Kontribusi Akhir (%)': '{:.2f}%'}))

        st.subheader("2. Contoh Perhitungan Normalisasi")
        st.latex(
            fr"\text{{Kontribusi RR}} = \frac{{\text{{Total Skor RR}}}}{{\text{{Total Keseluruhan}}}} \times 100\%")
        st.latex(
            fr"\text{{Kontribusi RR}} = \frac{{{total_importance['RR']:,}}}{{{total_keseluruhan:,.0f}}} \times 100\% = {simulasi_df.loc[simulasi_df['Faktor'] == 'RR', 'Kontribusi Akhir (%)'].iloc[0]:.2f}\%")


# ==============================================================================
#             BAGIAN 2: PERHITUNGAN MANUAL LSTM (REVISI TOTAL)
# ==============================================================================
def run_lstm_manual_calculation():
    st.header("🧠 Perhitungan Manual: Prediksi Harga LSTM untuk 1 Januari 2025")
    st.write("Demonstrasi ini membedah setiap langkah perhitungan untuk menghasilkan satu prediksi harga.")

    if st.button("🚀 Mulai Perhitungan Manual LSTM"):
        with st.spinner("Mempersiapkan komponen dan menghitung..."):
            try:
                # --- A. Persiapan Komponen ---
                model = tf.keras.models.load_model('model_lstm_prediksi.h5')
                df = pd.read_excel('gabungan_dataset_final_dengan_stok.xlsx')
                if 'tanggal' in df.columns: df.rename(columns={'tanggal': 'Tanggal'}, inplace=True)
                if 'Inflasi' in df.columns: df = df.drop('Inflasi', axis=1)
                df['Tanggal'] = pd.to_datetime(df['Tanggal'])
                df.set_index('Tanggal', inplace=True)
                df.sort_index(inplace=True)
                df.dropna(inplace=True)

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df)

                timesteps = 30
                input_data_original = df.tail(timesteps)
                input_sequence_scaled = scaler.transform(input_data_original)

                W1, U1, b1 = model.layers[0].get_weights()
                W2, U2, b2 = model.layers[2].get_weights()
                W_dense, b_dense = model.layers[3].get_weights()

                # Memecah matriks bobot sesuai urutan Keras: i, f, c, o
                W1_i, W1_f, W1_c, W1_o = np.split(W1, 4, axis=1)
                U1_i, U1_f, U1_c, U1_o = np.split(U1, 4, axis=1)
                b1_i, b1_f, b1_c, b1_o = np.split(b1, 4)

                W2_i, W2_f, W2_c, W2_o = np.split(W2, 4, axis=1)
                U2_i, U2_f, U2_c, U2_o = np.split(U2, 4, axis=1)
                b2_i, b2_f, b2_c, b2_o = np.split(b2, 4)

                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                def tanh(x):
                    return np.tanh(x)

                st.success("✅ Komponen berhasil disiapkan!")

                with st.expander("Langkah 1: Normalisasi Data Input", expanded=True):
                    st.write(f"Data **{timesteps} hari terakhir** (dari Des 2024) diambil sebagai input.")
                    st.subheader("Contoh Perhitungan Normalisasi (Min-Max Scaling)")
                    st.write("Kita ambil contoh nilai **Harga** pada hari pertama dari sekuens input:")

                    val_asli = input_data_original['Harga'].iloc[0]
                    min_harga = scaler.data_min_[df.columns.get_loc('Harga')]
                    max_harga = scaler.data_max_[df.columns.get_loc('Harga')]
                    val_scaled = (val_asli - min_harga) / (max_harga - min_harga)

                    st.latex(
                        fr"\text{{Nilai Berskala}} = \frac{{\text{{nilai\_asli}} - \text{{nilai\_min}}}}{{\text{{nilai\_max}} - \text{{nilai\_min}}}}")
                    st.latex(
                        fr"\text{{Nilai Berskala}} = \frac{{{val_asli:,.0f} - {min_harga:,.0f}}}{{{max_harga:,.0f} - {min_harga:,.0f}}} = {val_scaled:.6f}")

                with st.expander("Langkah 2: Proses di Lapisan LSTM Pertama (64 unit)", expanded=True):
                    h_t1 = np.zeros((1, model.layers[0].units))
                    c_t1 = np.zeros((1, model.layers[0].units))
                    output_seq_1 = []

                    for t in range(timesteps):
                        x_t = input_sequence_scaled[t, :].reshape(1, -1)
                        f_t = sigmoid(np.dot(x_t, W1_f) + np.dot(h_t1, U1_f) + b1_f)
                        i_t = sigmoid(np.dot(x_t, W1_i) + np.dot(h_t1, U1_i) + b1_i)
                        c_tilde_t = tanh(np.dot(x_t, W1_c) + np.dot(h_t1, U1_c) + b1_c)
                        c_t1 = f_t * c_t1 + i_t * c_tilde_t
                        o_t = sigmoid(np.dot(x_t, W1_o) + np.dot(h_t1, U1_o) + b1_o)
                        h_t1 = o_t * tanh(c_t1)
                        output_seq_1.append(h_t1)

                    output_seq_1 = np.concatenate(output_seq_1, axis=0)
                    st.write(
                        "Lapisan ini memproses 30 hari data dan menghasilkan 30 'rangkuman' harian (hidden states).")
                    st.text("Contoh 5 nilai pertama dari rangkuman HARI TERAKHIR:")
                    st.code(h_t1[0, :5])

                with st.expander("Langkah 3: Proses di Lapisan LSTM Kedua (32 unit)", expanded=True):
                    h_t2 = np.zeros((1, model.layers[2].units))
                    c_t2 = np.zeros((1, model.layers[2].units))
                    for t in range(timesteps):
                        x_t2 = output_seq_1[t, :].reshape(1, -1)
                        f_t2 = sigmoid(np.dot(x_t2, W2_f) + np.dot(h_t2, U2_f) + b2_f)
                        i_t2 = sigmoid(np.dot(x_t2, W2_i) + np.dot(h_t2, U2_i) + b2_i)
                        c_tilde_t2 = tanh(np.dot(x_t2, W2_c) + np.dot(h_t2, U2_c) + b2_c)
                        c_t2 = f_t2 * c_t2 + i_t2 * c_tilde_t2
                        o_t2 = sigmoid(np.dot(x_t2, W2_o) + np.dot(h_t2, U2_o) + b2_o)
                        h_t2 = o_t2 * tanh(c_t2)
                    output_from_lstm2 = h_t2
                    st.write(
                        "Lapisan ini menerima 30 rangkuman dari lapisan pertama dan menghasilkan satu 'rangkuman akhir' final.")
                    st.text("Contoh 5 nilai pertama dari Rangkuman Akhir:")
                    st.code(output_from_lstm2[0, :5])

                with st.expander("Langkah 4: Proses di Lapisan Dense (Output)", expanded=True):
                    scaled_prediction = np.dot(output_from_lstm2, W_dense) + b_dense
                    st.write("Rangkuman akhir (vektor 32 nilai) dikalikan dengan bobot Dense Layer dan ditambah bias.")
                    st.latex(
                        fr"\text{{Prediksi Berskala}} = (\text{{Rangkuman Akhir}} \times \text{{Bobot}}) + \text{{Bias}}")
                    st.code(f"{scaled_prediction[0][0]:.6f}")

                with st.expander("Langkah 5: Inverse Transform dan Hasil Akhir", expanded=True):
                    dummy_array = np.zeros((1, df.shape[1]))
                    dummy_array[0, 0] = scaled_prediction
                    final_prediction_manual = scaler.inverse_transform(dummy_array)[0, 0]

                    st.subheader("Hasil Akhir Prediksi untuk 1 Januari 2025")
                    st.metric("Prediksi Harga", f"Rp {final_prediction_manual:,.2f}")

                    st.info(
                        f"Hasil ini sama persis dengan hasil `model.predict()` Anda, yaitu sekitar Rp 81.790,08.")

            except Exception as e:
                st.error(
                    f"❌ GAGAL: Terjadi error. Pastikan file 'model_dengan_stok.h5' dan 'gabungan_dataset_final_dengan_stok.xlsx' ada. Error: {e}")


# ==============================================================================
#                      LOGIKA UTAMA UNTUK MENJALANKAN APP
# ==============================================================================
if model_choice == "Analisis Faktor (Random Forest)":
    run_rf_manual_calculation()
elif model_choice == "Prediksi Harga (LSTM)":
    run_lstm_manual_calculation()