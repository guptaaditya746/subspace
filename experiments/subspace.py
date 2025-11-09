import os
import pickle
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from tensorflow import keras
import mlflow
from loguru import logger
import sys

# Add project root to path to allow importing logging_config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from logging_config import configure_logger

from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, store_partial_cfs
from experiments.results.results_concatenator import concatenate_result_files

from methods.SubSpaCECF import SubSpaCECF

# DATASETS = ['CBF', 'chinatown', 'coffee', 'gunpoint', 'ECG200']
DATASETS = ['ECG200']
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 10


experiments = {
    'subspace': {
        'params': {
            'population_size': 100,
            'change_subseq_mutation_prob': 0.05,
            'elite_number': 4,
            'offsprings_number': 96,
            'max_iter': 100,
            'init_pct': 0.2,
            'reinit': True,
            'alpha': 0.2,
            'beta': 0.6,
            'eta': 0.2,
            'gamma': 0.25,
            'sparsity_balancer': 0.4,
        },
    },
}


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]
    nun_examples_worker = sample_dict["nun_examples"]
    desired_targets_worker = sample_dict["desired_targets"]

    # Get model
    model_worker = keras.models.load_model(f'models/{dataset}/{dataset}_best_model.hdf5')

    # Get outlier calculator
    with open(f'models/{dataset}/{dataset}_outlier_calculator.pickle', 'rb') as f:
        outlier_calculator_worker = pickle.load(f)

    # Instantiate the Counterfactual Explanation method
    cf_explainer = SubSpaCECF(model_worker, 'tf', outlier_calculator_worker, **params)

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig_worker = x_orig_samples_worker[i]
        nun_example_worker = nun_examples_worker[i]
        desired_target_worker = desired_targets_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig_worker, desired_target_worker, nun_example=nun_example_worker)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i+THREAD_SAMPLES-1,
                      dataset, file_suffix_name=exp_name)
    
    # Log partial result as artifact
    partial_result_path = f'./results/{dataset}/{exp_name}_{first_sample_i:04d}-{first_sample_i+THREAD_SAMPLES-1:04d}.pickle'
    mlflow.log_artifact(partial_result_path, artifact_path="partial_results")
    return 1


def experiment_dataset(dataset, exp_name, params):
    logger.info(f"Processing dataset: {dataset}")
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="./data")
    y_train, y_test = label_encoder(y_train, y_test)

    # Load model
    model = keras.models.load_model(f'models/{dataset}/{dataset}_best_model.hdf5')

    # Predict on x test
    y_pred_logits = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_logits, axis=1)

    logger.info("Retrieving NUNs for test set...")
    # Get the NUNs
    nuns_idx = []
    desired_classes = []
    for instance_idx in range(len(X_test)):
        distances, indexes, labels = nun_retrieval(X_test[instance_idx], y_pred[instance_idx],
                                                   'euclidean', 1, X_train, y_train)
        nuns_idx.append(indexes[0])
        desired_classes.append(labels[0])
    nuns_idx = np.array(nuns_idx)
    desired_classes = np.array(desired_classes)

    # START COUNTERFACTUAL GENERATION
    if MULTIPROCESSING:
        # Prepare dict to iterate optimization problem
        samples = []
        for i in range(I_START, len(X_test), THREAD_SAMPLES):
            # Init optimizer
            x_orig_samples = X_test[i:i + THREAD_SAMPLES]
            nun_examples = X_train[nuns_idx[i:i + THREAD_SAMPLES]]
            desired_targets = desired_classes[i:i + THREAD_SAMPLES]

            sample_dict = {
                "dataset": dataset,
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "nun_examples": nun_examples,
                "desired_targets": desired_targets,
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        logger.info('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

    # Concatenate the results
    logger.info("Concatenating partial results...")
    concatenate_result_files(dataset, exp_name)

    # Log final result as artifact
    final_result_path = f'./results/{dataset}/{exp_name}.pickle'
    if os.path.exists(final_result_path):
        mlflow.log_artifact(final_result_path, artifact_path="final_results")
        logger.info(f"Logged final result artifact: {final_result_path}")


if __name__ == "__main__":
    configure_logger()
    mlflow.set_experiment("subspace_experiments")

    for experiment_name, experiment_params in experiments.items():
        for dataset in DATASETS:
            with mlflow.start_run(run_name=f"{experiment_name}_{dataset}"):
                logger.info(f"Starting experiment '{experiment_name}' for dataset '{dataset}'...")
                mlflow.log_param("dataset", dataset)
                mlflow.log_param("experiment_name", experiment_name)
                mlflow.log_params(experiment_params["params"])
                experiment_dataset(
                    dataset,
                    experiment_name,
                    experiment_params["params"]
                )
    logger.info('Finished all experiments.')
