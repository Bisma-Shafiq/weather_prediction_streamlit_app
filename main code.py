import pandas as pd
from neuralprophet import NeuralProphet

# Data read
data = pd.read_csv('weatherAUS.csv')

# Preprocess data
Alb = data[data['Location'] == 'Albury']
Alb['Date'] = pd.to_datetime(Alb['Date'])

# Selecting and renaming columns
df = Alb[['Date', 'Temp9am']].copy()  # Selecting 'Date' and 'Temp9am' columns
df.dropna(inplace=True)  # Dropping rows with missing values
df.columns = ['ds', 'y']  # Renaming columns to 'ds' for datetime and 'y' for output (temperature)

# Train the Model
m = NeuralProphet()  # Instantiate NeuralProphet model
m.fit(df, freq='D', epochs=100)  # Fit the model with daily frequency and 100 epochs

# Forecast Model
future = m.make_future_dataframe(df, periods=100)
forecast = m.predict(future)

# Plot Forecast
fig_forecast = m.plot(forecast)
fig_forecast.show()
fig_forecast = m.plot_components(forecast)
fig_forecast.show()
# Explicitly display the Plotly plot
