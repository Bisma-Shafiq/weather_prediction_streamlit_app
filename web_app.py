import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet

def load_data():
    data = pd.read_csv('weatherAUS.csv')
    return data

def preprocess_data(data):
    Alb = data[data['Location'] == 'Albury']
    Alb['Date'] = pd.to_datetime(Alb['Date'])
    df = Alb[['Date', 'Temp9am']].copy()
    df.dropna(inplace=True)
    df.columns = ['ds', 'y']
    return df

def train_model(df):
    m = NeuralProphet()
    m.fit(df, freq='D', epochs=100)
    return m

def main():
    st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")

    # Set background color to blue
    st.markdown(
        """
        <style>
        body {
            background-color: #3498db;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Weather Prediction App')

    # Load and preprocess data
    data = load_data()
    df = preprocess_data(data)

    # Train the model
    model = train_model(df)

    # Slider for number of days to predict
    num_days = st.slider('Select the number of days to predict', min_value=1, max_value=365, value=100, step=1)

    # Forecast
    future = model.make_future_dataframe(df, periods=num_days)
    forecast = model.predict(future)

    # Plot forecast
    st.subheader('Forecast Plot')
    fig_forecast = model.plot(forecast)
    st.plotly_chart(fig_forecast)

    # Plot components
    st.subheader('Forecast Components')
    fig_components = model.plot_components(forecast)
    st.plotly_chart(fig_components)

if __name__ == '__main__':
    main()
