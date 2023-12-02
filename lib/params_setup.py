from config import tpa_lstm_params
from lib.utils import create_dir, check_path_exists
import json
import logging
import tensorflow as tf

def params_setup():
    para = tpa_lstm_params
    
    if para.data_set == "muse" or para.data_set == "lpd5":
        para.mts = 0
        
    if para.attention_len == -1:
        para.attention_len = para.max_len
    
    # saving the parameters of a model into a JSON file
    create_dir(para.model_dir)
    json_path = para.model_dir + '/parameters.json'
    json.dump(vars(para), open(json_path, 'w'), indent=4)
    return para

    
def logging_config_setup(para):
    # Determine the logging level for the whole application
    logging.basicConfig(
        level=para.logging_level, 
        format='%(levelname)-8s - %(message)s',
        filename=para.model_dir + '/progress.txt' if para.file_output != 0 else None
    )

    # If file output is enabled, also log to console
    if para.file_output != 0:
        logging.getLogger().addHandler(logging.StreamHandler())

    # Set the verbosity level for TensorFlow
    # In TensorFlow 2.x, this will adhere to the Python logging settings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    
def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config