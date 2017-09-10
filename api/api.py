from flask import Flask, render_template, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Expecting json to be an object with these fields:
#
# {
#   "model": "lr", "ransac", "rfr", "sgd", or "svr"
#   "observations": {
#     "x": Int from 0-9,
#     "y": Int from 0-9,
#     "month": "jan", "feb", "mar", "apr", "may", "jun", 
#              "jul", "aug", "sep", "oct", "nov", or "dec"
#     "day": "mon", "tue", "wed", "thu", "fri", "sat", or "sun"
#     "ffmc": Float,
#     "dmc": Float,
#     "dc": Float,
#     "isi": Float,
#     "temp": Float,
#     "rh": Float,
#     "wind": Float,
#     "rain": Float
#   }
# }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model = data['model']
    observations = list(data['observations'].values())

    print(model, observations)
    return

if __name__ == '__main__':
    app.run(debug=True)
