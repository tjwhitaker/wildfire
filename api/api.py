from flask import Flask, jsonify, render_template, request
from sklearn.externals import joblib
from pipeline import AttributeSelector, CustomBinarizer, FullPipeline
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Expecting json to be an object with these fields:
#
# {
#   "algo": "lr", "ransac", "rfr", "sgd", or "svr"
#   "observations": {
#     "X": Int from 0-9,
#     "Y": Int from 0-9,
#     "month": "jan", "feb", "mar", "apr", "may", "jun", 
#              "jul", "aug", "sep", "oct", "nov", or "dec"
#     "day": "mon", "tue", "wed", "thu", "fri", "sat", or "sun"
#     "FFMC": Float,
#     "DMC": Float,
#     "DC": Float,
#     "ISI": Float,
#     "temp": Float,
#     "RH": Float,
#     "wind": Float,
#     "rain": Float
#   }
# }

# Returns
# {
#   "area": Float
# }


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        algo = data['algo']
        observations = data['observations']
    else:
        algo = request.form['algo']
        observations = {
            "X": request.form['x'],
            "Y": request.form['y'],
            "month": request.form['month'], 
            "day": request.form['day'],
            "FFMC": request.form['ffmc'],
            "DMC": request.form['dmc'],
            "DC": request.form['dc'],
            "ISI": request.form['isi'],
            "temp": request.form['temp'],
            "RH": request.form['rh'],
            "wind": request.form['wind'],
            "rain": request.form['rain']
        }
    
    df = pd.DataFrame([observations], columns=observations.keys())    
    pipeline = FullPipeline()
    data_prepared = pipeline.prepare_data(df)    

    if algo == 'lr':
        model = joblib.load('models/lr_model.pkl')
    elif algo == 'ransac':
        model = joblib.load('models/ransac_model.pkl')
    elif algo == 'rfr':
        model = joblib.load('models/rfr_model.pkl')
    elif algo == 'sgd':
        model = joblib.load('models/sgd_model.pkl')
    else:
        model = joblib.load('models/svr_model.pkl')

    prediction = {'area': model.predict(data_prepared)[0]}
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
