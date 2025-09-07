from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):
    try:
        gestation = float(form_data.get('gestation', 0))
        parity = int(form_data.get('parity', 0))
        age = float(form_data.get('age', 0))
        height = float(form_data.get('height', 0))
        weight = float(form_data.get('weight', 0))
        smoke = int(form_data.get('smoke', 0))  # 0 or 1
    except (ValueError, TypeError):
        return None  # Invalid input values

    cleaned_data = {
        "gestation": gestation,
        "parity": parity,
        "age": age,
        "height": height,
        "weight": weight,
        "smoke": smoke
    }

    return cleaned_data


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def get_predictions():
    baby_data_form = request.form

    baby_data_cleaned = get_cleaned_data(baby_data_form)
    if baby_data_cleaned is None:
        return "Error: Invalid input values"

    baby_df = pd.DataFrame([baby_data_cleaned], columns=['gestation', 'parity', 'age', 'height', 'weight', 'smoke'])

    with open("model/model.pkl", 'rb') as obj:
        model = pickle.load(obj)

    prediction = model.predict(baby_df)
    prediction = round(float(prediction), 2)

    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
