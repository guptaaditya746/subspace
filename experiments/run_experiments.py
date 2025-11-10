#!/usr/bin/env python
# coding: utf-8

# --- Logger Setup (MUST BE FIRST) ---
import sys
import os
import pickle
import numpy as np

# Add the parent directory (project root) to the Python path
# This allows us to import 'logging_config' from the parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from logging_config import configure_logger
    from loguru import logger
    
    # Configure the logger, saving logs to the 'logs' folder in the project root
    configure_logger(log_dir=os.path.join(project_root, "logs"))
    
except ImportError:
    print("ERROR: Could not import logging_config.py. Make sure it is in the parent directory.")
    sys.exit(1)

logger.info(f"Starting experiment script in: {script_dir}")
# --- End Logger Setup ---


import tensorflow as tf
from tensorflow import keras
import warnings

# Import from local utils (now that logger is set up)
from experiment_utils import local_data_loader, label_encoder

# --- Import the new segregated method files ---
from native_guide import generate_native_guide_cfs
# We import from 'watcher.py' as per your filename
from wachter import generate_wachter_cfs

# Suppress warnings
warnings.filterwarnings('ignore')
logger.info(f"TensorFlow Version: {tf.__version__}")

# Set the datasets you want to run
datasets = ['CBF']

def load_all_data_and_models(dataset_list):
    """
    Loads all datasets and models into dictionaries.
    """
    data_dict = {}
    models_dict = {}
    
    for dataset in dataset_list:
        logger.info(f"Loading data for {dataset}...")
        X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="./data")
        y_train, y_test = label_encoder(y_train, y_test)
        
        # Add a channel dimension if data is 2D (samples, timesteps)
        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            
        data_dict[dataset] = (X_train, y_train, X_test, y_test)

        # Load model
        model_path = f'./models/{dataset}/{dataset}_best_model.hdf5'
        try:
            model = keras.models.load_model(model_path)
            models_dict[dataset] = model
        except (IOError, FileNotFoundError):
            logger.error(f"Model file not found at {model_path}")
            sys.exit(1)
            
    return data_dict, models_dict

def main():
    # 1. Load all data and models first
    logger.info("--- Loading Data and Models ---")
    data_dict, models_dict = load_all_data_and_models(datasets)
    logger.info("All data and models loaded.")
    
    # 2. Run Wachter et al.
    # logger.info("\n--- Generating Wachter et al. Counterfactuals ---")
    # for dataset in datasets:
    #     if dataset in data_dict and dataset in models_dict:
    #         # We pass the 'models_dict' which contains the loaded model
    #         generate_wachter_cfs(dataset, data_dict, models_dict[dataset])

    # 3. Run Native Guide
    logger.info("\n--- Generating Native Guide Counterfactuals ---")
    for dataset in datasets:
        if dataset in data_dict and dataset in models_dict:
            generate_native_guide_cfs(dataset, data_dict, models_dict)

    logger.info("\n--- All experiments complete. ---")

if __name__ == "__main__":
    main()