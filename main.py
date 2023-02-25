from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('mualijModel.h5')

@app.route('/')
def home():
    return render_template('default.html')

@app.route('/predict', methods=['POST'])
def mentalHealth_prediction():
    # if request.method == "POST":
    # worker_type =  date_arr = request.form["work_type"]
    data = request.get_json()
    dict_data = dict(data)
    print(dict_data)

    # Age
    age = dict_data["age"]

    # Gender
    if dict_data["gender"] == 'female':
        gender = 0
    elif dict_data["gender"] == 'male':
        gender = 1
    else:
        gender = 2

    # Family History
    if dict_data["family_history"] == 'No':
        family_history = 0
    else:
        family_history = 1

    # Benefits
    if dict_data["benefits"] == "Dont Know":
        benefits = 0
    elif dict_data["benefits"] == "No":
        benefits = 1
    else:
        benefits = 2

    # Care Options
    if dict_data["care_options"] == 'No':
        care_options = 0
    elif dict_data["care_options"] == 'Not Sure':
        care_options = 1
    else:
        care_options = 2

    # Anonymity
    if dict_data["anonymity"] == "Dont know":
        anonymity = 0
    elif dict_data["anonymity"] == "No":
        anonymity = 1
    else:
        anonymity = 2

    # Leave
    if dict_data["leave"] == "Dont Know":
        leave = 0
    elif dict_data["leave"] == "Somewhat difficult":
        leave = 1
    elif dict_data["leave"] == "Somewhat easy":
        leave = 2
    elif dict_data["leave"] == "Very difficult":
        leave = 3
    else:
        leave = 4

    # Work Interfere
    if dict_data["work_interfere"] == "Dont Know":
        work_interfere = 0
    elif dict_data["work_interfere"] == "Never":
        work_interfere = 1
    elif dict_data["work_interfere"] == "Often":
        work_interfere = 2
    elif dict_data["work_interfere"] == "Rarely":
        work_interfere = 3
    else:
        work_interfere = 4

    inp = np.array([[age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere]])
    print(inp)
    result = model.predict(inp)
    result = [1 if y >= 0.5 else 0 for y in result]
    print(result)

    return {"status": "200", "prediction": result[0]}

if __name__ == "__main__":
    #predict()
    app.run(debug=True)