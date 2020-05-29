import logging
import pandas as pd
import requests
import zmq
import urllib.parse as urlparse
from cyber_sense_brain import dnn_predictor_online_pkg as dnn
import http.server as SimpleHTTPServer
import socketserver as SocketServer
from datetime import datetime
import numpy as np

SIMULATED_DATA_URL = "http://localhost:8000/NeuLogAPI?start=0&end=600"  # TODO: Change the start and end
# SERVER CONFIG
TOTAL_MIN_IN_VR = 10
GET_DATA_REQ = "GetPrediction"
time_step = 60
number_of_features = 8
normalizer_model = "dnn-models/normalizer.save"
dnn_model_file = "dnn-models/lstm_60t_120b"
PORT = 5555

# Set up log info
logging.basicConfig(filename='network.log', filemode='a',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def get_simulated_data():
    response = requests.get(url=SIMULATED_DATA_URL)
    if response.ok:
        json_data = response.json()
        print("Data Received...")
        data = prepare_data_frame(json_data)
        return data

    else:
        print("Cannot get Data")
        return False


def prepare_data_frame(json_data):
    data = {"HR": json_data["HR"], "PC_HR": json_data["PC_HR"], "HR_MIN": json_data["HR_MIN"],
            "HR_MAX": json_data["HR_MAX"],
            # "HRV": json_data["HRV"], "PC_HRV": json_data["PC_HRV"],
            # "HRV_MIN": json_data["HRV_MIN"], "HRV_MAX": json_data["HRV_MAX"], "PC_BR": json_data["PC_BR"],
            "GSR": json_data["GSR"], "PC_GSR": json_data["PC_GSR"], "GSR_MIN": json_data["GSR_MIN"],
            "GSR_MAX": json_data["GSR_MAX"]}

    return data


def predict_cybersickness(minute_wise_data, model):
    t = datetime.now()
    start_time = t.strftime('%m-%d-%Y-%H-%M-%S')
    resting_data = pd.DataFrame(columns=["GSR", "HR"])

    # TODO: split the minute wise data for x and y(target)
    normalized_data = dnn.normalize_test_data(minute_wise_data)
    test_data = dnn.prepare_timestep_data(normalized_data, time_step, 1)
    test_data.to_csv("results/test_data-" + start_time + "-simulation.csv")
    test_data = dnn.reshape_test_data(test_data)
    predicted_y = dnn.make_prediction(model, test_data, normalizer_model)
    np.savetxt("results/predicted.csv", predicted_y, delimiter=",")  # TODO: Make sure it does not ovverride
    logging.info("Predicted Result Saved.")
    avg_sick_score = np.average(predicted_y)
    print("Average Sickness Score: ", avg_sick_score)

    return str(avg_sick_score)


def request_data_from_sensors():
    logging.info("Requesting Data from Physiological Sensors...")
    minute_wise_data = get_simulated_data()
    logging.info("Successfully got data from sensors.")
    return pd.DataFrame(minute_wise_data)


class GetHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    # http://localhost:5555/GetPrediction
    def do_GET(self):
        self._set_headers()
        print("Get Request Received...")
        print("Command: ", self.command)
        print("Path: ", self.path)
        if GET_DATA_REQ in self.path:
            data = request_data_from_sensors()
            cyber_sickness_prediction = predict_cybersickness(data, loaded_model)
            self.wfile.write(cyber_sickness_prediction.encode("utf8"))


def start_cyber_sense():
    Handler = GetHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Starting CyberSense Server...")
    print("Waiting for VR Roller Coaster to start...")
    httpd.serve_forever()


# Run Experiment
if __name__ == '__main__':
    loaded_model = dnn.load_model_from_file(dnn_model_file)
    start_cyber_sense()
