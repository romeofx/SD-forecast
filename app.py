from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import joblib
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime

app = Flask(__name__)

# Load saved Prophet model
model = joblib.load("prophet_model.pkl")  # Adjust name if different

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get user input
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Create future dataframe
        future_dates = pd.date_range(start=start_date, end=end_date)
        future_df = pd.DataFrame({'ds': future_dates})

        # Predict
        forecast = model.predict(future_df)

        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dot')))
        graph_html = pio.to_html(fig, full_html=False)

        # Return table and plot
        return render_template('forecast.html', graph_html=graph_html, table=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2).to_html(classes='table table-striped'))
    
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
