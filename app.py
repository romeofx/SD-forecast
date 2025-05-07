from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import joblib
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime

app = Flask(__name__)

# Load saved Prophet model (trained with external factors, if applicable)
model = joblib.load("prophet_model.pkl") 

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

        # TODO: In the future, collect user inputs for external regressors (weather, tariffs, demand) here.

        # Predict
        forecast = model.predict(future_df)

        # Fix negative forecast values (e.g., export values can't be negative)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
        forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
        forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))

        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
        graph_html = pio.to_html(fig, full_html=False)

        # Rename columns for clarity
        forecast_table = forecast.rename(columns={
            'ds': 'Date',
            'yhat': 'Export Value',
            'yhat_lower': 'Export Value (Lower)',
            'yhat_upper': 'Export Value (Upper)'
        })

        # Return table and plot
        return render_template(
            'forecast.html',
            graph_html=graph_html,
            table=forecast_table[['Date', 'Export Value', 'Export Value (Lower)', 'Export Value (Upper)']].round(2).to_html(classes='table table-striped', index=False)
        )

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)

