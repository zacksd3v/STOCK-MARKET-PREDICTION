import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.sidebar.title("🎓 FUDMA 2026")
st.sidebar.markdown("---")
st.sidebar.write(f"**Name:** Suleiman Jafar Danyaro")
st.sidebar.write(f"**Matric No:** CSA/2023/27615")
st.sidebar.write(f"**Faculty:** Computing")
st.sidebar.write(f"**Department:** CS & IT")
st.sidebar.write(f"**Topic:** Developing a Robust Algorithmic Framework for Autonomous Stock Price Prediction")
st.sidebar.markdown("---")
st.sidebar.info("This project uses LSTM Deep Learning to predict stock prices based on historical data.")

st.title('📈 Stock Market Predictor (2019 - 2026)')
st.markdown("Developed by: **Suleiman Jafar Danyaro**")

try:
    model = load_model('Stock Price Prediction.keras')
except Exception as e:
    st.error("Error: Could not load the model file. Make sure 'Stock Price Prediction.keras' is in the folder.")
    st.stop() # Stop the execution if model is not found

stock = st.text_input('Enter Stock Symbol e.g (AAPL, MSFT, GOOGL, AMZN, NVDA)', 'GOOG')

if not stock:
    st.warning("Please enter a stock symbol to begin e.g NVDA.")
else:
    start = '2019-01-01'
    end = '2026-12-31'

    try:
        with st.spinner(f'Fetching data for {stock}...'):
            data = yf.download(stock, start, end)

        # 1. Check if data is empty (Wrong symbol or no internet)
        if data.empty:
            st.error(f"No data found for symbol '{stock}'. Please check the symbol and try again.")
        else:
            # Fix for Multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # PRICE METRICS ---
            try:
                last_price = float(data['Close'].iloc[-1])
                prev_price = float(data['Close'].iloc[-2])
                change = float(last_price - prev_price)
                
                col1, col2 = st.columns(2)
                col1.metric(label=f"Current {stock} Price", value=f"${last_price:.2f}", delta=f"{change:.2f}")
                col2.write("Data Range: Jan 2019 - Present")
            except Exception:
                st.info("Could not calculate daily change metrics.")

            st.subheader('Stock Data Summary')
            st.write(data.tail(10))

            try:
                data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
                data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

                scaler = MinMaxScaler(feature_range=(0,1))
                pas_100_days = data_train.tail(100)
                data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
                data_test_scale = scaler.fit_transform(data_test)

                with st.spinner('Generating Graphs and Predictions...'):
                    st.subheader('Price vs Moving Averages')
                    ma_50_days = data.Close.rolling(50).mean()
                    ma_100_days = data.Close.rolling(100).mean()
                    ma_200_days = data.Close.rolling(200).mean()

                    # Combining graphs for better presentation
                    fig1, ax1 = plt.subplots(figsize=(10,6))
                    ax1.plot(data.Close, 'g', label='Original Price')
                    ax1.plot(ma_50_days, 'r', label='MA50')
                    ax1.plot(ma_100_days, 'b', label='MA100')
                    ax1.plot(ma_200_days, 'orange', label='MA200')
                    ax1.legend()
                    st.pyplot(fig1)

                    # PREDICTION LOGIC ---
                    x = []
                    for i in range(100, data_test_scale.shape[0]):
                        x.append(data_test_scale[i-100:i])

                    x = np.array(x)
                    if x.size == 0:
                        st.error("Not enough data to make a prediction. Try a stock with a longer history.")
                    else:
                        predict = model.predict(x)
                        scale = 1/scaler.scale_[0]
                        predict = predict * scale
                        actual_test_data = data_test.iloc[100:].values
                        
                        st.subheader('Original Price vs Predicted Price')
                        fig4 = plt.figure(figsize=(10,6))
                        plt.plot(actual_test_data, 'g', label='Original Price')
                        plt.plot(predict, 'r', label='Predicted Price')
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend()
                        st.pyplot(fig4)
            except Exception as e:
                st.error(f"An error occurred during data processing: {e}")

    except Exception as e:
        st.error(f"Connection Error: Please check your internet connection. ({e})")

st.markdown("---")
st.caption(f"© 2026 | {stock if stock else 'Stock'} Price Prediction System | Developed by Suleiman Jafar Danyaro")