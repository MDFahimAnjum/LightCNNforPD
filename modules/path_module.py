import os

current_directory = os.getcwd() #parameters

processed_datapath = os.path.join(current_directory,"data/processed") # Define the path of processed data
cnn_modelpath=os.path.join(current_directory,"models")
results_path=os.path.join(current_directory,"results")
