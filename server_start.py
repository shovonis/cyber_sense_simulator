import datetime
import logging
import logging.handlers
import time
import numpy as np
import pandas as pd
import requests
import zmq
import data_manager as dp
from cyber_sense_brain import dnn_predictor_online_pkg as dnn
from datetime import datetime

# NeuLog API Endpoint
GetValueURL = "http://localhost:22002/NeuLogAPI?GetSensorValue:[GSR],[1],[Pulse],[1]"
START_EXP_URL = "http://localhost:22002/NeuLogAPI?StartExperiment:[GSR],[1],[Pulse],[1]"
GET_EXP_SAMPL_URL = "http://localhost:22002/NeuLogAPI?GetExperimentSamples"
STOP_EXP_URL = "http://localhost:22002/NeuLogAPI?StopExperiment"
SIMULATED_DATA_URL = "http://localhost:8000/NeuLogAPI?start=0&end=60"

WAIT_TIME = 62  # start a new experiment in every 60 second
NUMBER_OF_SAMPLES = 600  # Total Samples in a minute
HR = 1
GSR = 0

logging.basicConfig(filename='network.log', filemode='a',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def get_experiment_data():
    response = requests.get(url=GET_EXP_SAMPL_URL)
    if response.ok:
        return response.json()
    else:
        print("Cannot get Data")
        return False


def get_simulated_data():
    response = requests.get(url=SIMULATED_DATA_URL)
    if response.ok:
        print(response.json())
    else:
        print("Cannot get Data")
        return False


def get_resting_data_five_mins():
    logging.info("Resting individual_raw_data collection started.")
    json_data = get_experiment_data()
    GSR_data = json_data["GetExperimentSamples"][GSR][2:]
    HR_data = json_data["GetExperimentSamples"][HR][2:]
    data = {"GSR": GSR_data, "HR": HR_data}

    logging.info("Resting individual_raw_data collected")
    return data


def get_minute_wise_data():
    json_data = get_experiment_data()
    GSR_data = json_data["GetExperimentSamples"][GSR][2:]
    HR_data = json_data["GetExperimentSamples"][HR][2:]
    data = {"GSR": GSR_data, "HR": HR_data}

    return data


def start_experiment(data_col_freq, num_of_samples):
    stop_experiment()  # Stop previous experiment
    response = requests.get(url=START_EXP_URL + ",[" + str(data_col_freq) + "],[" + str(num_of_samples) + "]")
    if response.ok:
        return response.json()
    else:
        print("ERROR!! Cannot start experiment")
        return False


def stop_experiment():
    response = requests.get(url=STOP_EXP_URL)
    if response.ok:
        return response.json()
    else:
        print("ERROR!! Cannot stop experiment")
        return False


def start_exp():
    stop_experiment()
    print("Waiting for VR to start...")
    loaded_model = dnn.load_model_from_file("resource/1_lstm_60t_128b")
    resting_data = pd.DataFrame(columns=["GSR", "HR"])
    while True:
        message = socket.recv()
        print("Received request :%s" % message)
        message = message.decode("utf-8")
        print("Experiment Name: ", message)

        logging.info("#############Experiment Started##############")

        if message in SIMULATION:
            logging.info("Started simulation experiment..")
            t = datetime.now()
            start_time = t.strftime('%m-%d-%Y-%H-%M-%S')
            minute_wise_data = get_minute_wise_data()
            # TODO: split the minute wise data for x and y(target)
            normalized_data = dnn.normalize_test_data(minute_wise_data)
            test_data = dnn.prepare_timestep_data(normalized_data, time_step, 1)
            test_data.to_csv("data/minute/test_data-" + start_time + "-simulation.csv")
            test_data = dnn.reshape_test_data(test_data)
            predicted_y = dnn.make_prediction(loaded_model, test_data, normalizer_model)
            np.savetxt("results/minute/predicted.csv", predicted_y,
                       delimiter=",")  # TODO: Make sure it does not ovverride
            print("Average Sickness Score: ", np.average(predicted_y))
            logging.info("Simulation Data Saved.")
            stop_experiment()
            print("Send: Done")
            logging.info("Successfully run experiment. Sending notification to VR...")
            socket.send(b"Done")


####################### SERVER ####################

TOTAL_MIN_IN_VR = 10
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
data_freq = 3
counter = 0

RESTING_EXP_TIME = 60
WAIT_TIME_FOR_SENDING_DATA_REQ = 60
NUM_OF_TIMES_DATA_REQ_SEND = 5
RESTING = "REST"
SIMULATION = "SIM"
DATA_FREQ_CODE = 8
NUMBER_OF_SAMPLES = 600

time_step = 60
number_of_features = 8
normalizer_model = "resource/normalizer.save"
## Run Experiment ###
start_exp()
