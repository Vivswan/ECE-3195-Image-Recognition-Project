from run_assignment_p1 import assignment_p1
from run_assignment_p2 import assignment_p2, P2_PARAMETERS
from src.path_functions import get_relative_path

DEVICE_NAME = "cuda:0"
DATA_FOLDER = "data1"

if __name__ == '__main__':
    assignment_p1(data_folder=DATA_FOLDER, device_name=DEVICE_NAME)
    for i in P2_PARAMETERS:
        parameters = P2_PARAMETERS[i]
        parameters["device_name"] = DEVICE_NAME
        parameters["data_folder"] = get_relative_path(__file__, DATA_FOLDER)
        assignment_p2(kwargs=parameters, **parameters)
