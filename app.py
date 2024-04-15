from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submissionn
        try:
            gdp = float(request.form['gdp'])
            social_support = float(request.form['social_support'])
            life_expectancy = float(request.form['life_expectancy'])
            freedom = float(request.form['freedom'])
            generosity = float(request.form['generosity'])
            corruption = float(request.form['corruption'])

            model = joblib.load('linear_regression_model.pkl')
            input_data = np.array([[gdp, social_support, life_expectancy, freedom, generosity, corruption]], dtype=float)
            predicted_score = model.predict(input_data)[0]

            return render_template('result.html', score=predicted_score)
        except Exception as e:
            # Handle any errors gracefully
            error_message = str(e)
            return render_template('error.html', error=error_message)
    # If it's not a POST request, render index.html
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
