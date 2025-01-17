from flask import Flask, request, render_template_string
import pickle
import json
import numpy as np

__model = None
__data_columns = None

def load_saved_artifacts():
    global __data_columns
    global __model

    # Load feature columns
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts2\\columns_new.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']

    # Load the trained model
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts2\\Gradient Boosting_nautical_mile_final.pkl", 'rb') as model_file:
        __model = pickle.load(model_file)
    print("Artifacts loaded successfully.")


EXPECTED_FEATURES=[
    'airpressure', 'airtemperature', 'averagespeedgps', 'averagespeedlog',
    'cargometrictons', 'currentstrength', 'distancefromlastport',
    'distancetonextport', 'distancetravelledsincelastreport',
    'enginedriftingstoppagetime', 'engineroomairpressure',
    'engineroomairtemperature', 'engineroomrelativeairhumidity',
    'engineslip', 'isfuelchangeover', 'isturbochargercutout',
    'relativeairhumidity', 'remainingdistancetoeosp', 'remainingtimetoeosp',
    'scavengingaircoolingwatertemperatureaftercooler', 'scavengingairpressure',
    'scavengingairtemperatureaftercooler', 'seastate', 'seastatedirection',
    'totalcylinderoilconsumption', 'totalcylinderoilspecificconsumption',
    'watertemperature', 'winddirection', 'winddirectionisvariable',
    'tugsused', 'voyagenumber', 'distanceeosptofwe', 'timesteamed',
    'bendingmomentsinpercent', 'dischargedsludge', 'metacentricheight',
    'shearforcesinpercent', 'distancetoeosp', 'saileddistance',
    'runninghourscountervalue', 'energyproducedcountervalue',
    'energyproducedinreportperiod', 'consumption', 'runninghours',
    'new_timezoneinfo_05:30', 'new_timezoneinfo_07:30',
    'new_timezoneinfo_08:30', 'new_timezoneinfo_09:30',
    'new_timezoneinfo_10:30', 'new_timezoneinfo_11:00',
    'new_timezoneinfo_11:30', 'new_timezoneinfo_12:00',
    'new_timezoneinfo_12:30', 'new_timezoneinfo_13:30',
    'new_timezoneinfo_14:30', 'new_timezoneinfo_15:30',
    'new_timezoneinfo_16:30', 'new_timezoneinfo_17:30',
    'new_timezoneinfo_3:30', 'new_timezoneinfo_4:30',
    'new_timezoneinfo_5:30', 'new_timezoneinfo_6:30',
    'new_timezoneinfo_7:30', 'totalconsumption'
]

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance):
    # global EXPECTED_FEATURES
    x = np.zeros(len(EXPECTED_FEATURES))
    x[0] = airpressure
    x[42] = consumption
    x[24] = totalcylinderoilconsumption
    x[25] = totalcylinderoilspecificconsumption
    x[38] = saileddistance
    # Predict fuel consumption
    fuel_per_nautical_mile = __model.predict([x])[0]
    total_consumption = fuel_per_nautical_mile * saileddistance
    return fuel_per_nautical_mile, total_consumption

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fuel Consumption Prediction API</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <h1 class="mt-5">Fuel Consumption Prediction API</h1>
                <form action="/predict" method="post" class="mt-3">
                    <div class="form-group">
                        <label for="airpressure">Air Pressure:</label>
                        <input type="text" class="form-control" id="airpressure" name="airpressure" required>
                    </div>
                    <div class="form-group">
                        <label for="consumption">Consumption:</label>
                        <input type="text" class="form-control" id="consumption" name="consumption" required>
                    </div>
                    <div class="form-group">
                        <label for="totalcylinderoilconsumption">Total Cylinder Oil Consumption:</label>
                        <input type="text" class="form-control" id="totalcylinderoilconsumption" name="totalcylinderoilconsumption" required>
                    </div>
                    <div class="form-group">
                        <label for="totalcylinderoilspecificconsumption">Total Cylinder Oil Specific Consumption:</label>
                        <input type="text" class="form-control" id="totalcylinderoilspecificconsumption" name="totalcylinderoilspecificconsumption" required>
                    </div>
                    <div class="form-group">
                        <label for="saileddistance">Sailed Distance:</label>
                        <input type="text" class="form-control" id="saileddistance" name="saileddistance" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Predict</button>
                </form>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.6.0/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
    """)


@app.route('/predict', methods=["POST"])
def predict():
    airpressure = float(request.form['airpressure'])
    consumption = float(request.form['consumption'])
    totalcylinderoilconsumption = float(request.form['totalcylinderoilconsumption'])
    totalcylinderoilspecificconsumption = float(request.form['totalcylinderoilspecificconsumption'])
    saileddistance = float(request.form['saileddistance'])

    fuel_per_nautical_mile, total_consumption = estimate_fuel_consumption(
        airpressure, consumption, totalcylinderoilconsumption,
        totalcylinderoilspecificconsumption, saileddistance
    )

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fuel Consumption Prediction Result</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center">Fuel Consumption Prediction Results</h1>
                <div class="card mt-4">
                    <div class="card-body">
                        <p><strong>Air Pressure:</strong> {{ airpressure }}</p>
                        <p><strong>Consumption:</strong> {{ consumption }}</p>
                        <p><strong>Total Cylinder Oil Consumption:</strong> {{ totalcylinderoilconsumption }}</p>
                        <p><strong>Total Cylinder Oil Specific Consumption:</strong> {{ totalcylinderoilspecificconsumption }}</p>
                        <p><strong>Sailed Distance:</strong> {{ saileddistance }}</p>
                        <h2 class="mt-4"><strong>Total Consumption:</strong> {{ total_consumption }}</h2>
                        <h2><strong>Fuel Per Nautical Mile:</strong> {{ fuel_per_nautical_mile }}</h2>
                    </div>
                </div>
                <div class="text-center mt-3">
                    <a href="/" class="btn btn-primary">Back to Home</a>
                </div>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.6.0/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.amazonaws.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
    """, airpressure=airpressure, consumption=consumption, totalcylinderoilconsumption=totalcylinderoilconsumption, totalcylinderoilspecificconsumption=totalcylinderoilspecificconsumption, saileddistance=saileddistance, total_consumption=total_consumption, fuel_per_nautical_mile=fuel_per_nautical_mile)

if __name__ == '__main__':
    try:
        load_saved_artifacts()
        print("Server is running.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        exit(1)
    app.run(debug=True, port=5001)



