from bottle import post, request, route, run, static_file, template
from sklearn.externals import joblib
from pipeline import AttributeSelector, CustomBinarizer, FullPipeline
import pandas as pd

@route('/')
def index():
    return template('templates/index.html')

@post('/predict')
def predict():
    if request.json:
        data = request.json()
        algo = data['algo']
        observations = data['observations']
    else:
        algo = request.forms.get('algo')
        observations = {
            "X": request.forms.get('x'),
            "Y": request.forms.get('y'),
            "month": request.forms.get('month'), 
            "day": request.forms.get('day'),
            "FFMC": request.forms.get('ffmc'),
            "DMC": request.forms.get('dmc'),
            "DC": request.forms.get('dc'),
            "ISI": request.forms.get('isi'),
            "temp": request.forms.get('temp'),
            "RH": request.forms.get('rh'),
            "wind": request.forms.get('wind'),
            "rain": request.forms.get('rain')
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
    
    return prediction

run(host='localhost', port=8080, reloader=True)
