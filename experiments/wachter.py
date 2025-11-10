#!/usr/bin/env python
# coding: utf-8

# Contains all logic for the Wachter et al. counterfactual method.
# This version is optimized for multiprocessing (MPU) and logging.

import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy import stats
import multiprocessing
from loguru import logger # Import the logger

# --- Helper Functions (Must be at top-level) ---

def _target_label(label):
    return 1 if label == 0 else 0

def _dist_mad(query, cf, X_train_mad_scalar):
    manhat = np.abs(query - cf)
    if X_train_mad_scalar == 0:
        X_train_mad_scalar = 1e-8 
    return np.sum((manhat / X_train_mad_scalar).flatten())


class WachterGenerator:
    """
    Class-based generator, instantiated *inside* a worker process.
    """
    def __init__(self, model, X_train_mad_scalar, ts_length, n_features, lambda_init=0.1, max_iter=500, pred_threshold=0.5):
        self.model = model
        self.X_train_mad_scalar = X_train_mad_scalar
        self.ts_length = ts_length
        self.n_features = n_features
        self.lambda_init = lambda_init
        self.max_iter = max_iter
        self.pred_threshold = pred_threshold
        self.query = None
        self.target_label = None
        self.lamda = None

    def _loss_function(self, x_dash):
        x_dash_reshaped = x_dash.reshape(1, self.ts_length, self.n_features)
        pred_prob = self.model.predict(x_dash_reshaped, verbose=0)[0][self.target_label]
        pred_loss = (pred_prob - 1)**2
        dist_loss = _dist_mad(self.query, x_dash_reshaped, self.X_train_mad_scalar)
        return self.lamda * pred_loss + dist_loss

    def generate(self, instance, instance_label):
        self.lamda = self.lambda_init
        self.query = instance.reshape(1, self.ts_length, self.n_features)
        self.target_label = _target_label(instance_label)
        
        x0 = self.query.flatten() 
        
        res = minimize(self._loss_function, x0, method='nelder-mead', options={'maxiter': 10, 'xatol': 50, 'adaptive': True})
        cf = res.x.reshape(1, self.ts_length, self.n_features)
        prob_target = self.model.predict(cf, verbose=0)[0][self.target_label]
        
        i = 0
        while prob_target < self.pred_threshold:
            i += 1
            if i == self.max_iter:
                # Log a warning if it fails to converge
                logger.warning(f"Could not find CF after {self.max_iter} iterations. Returning last attempt.")
                break
            
            self.lamda = self.lambda_init * (1 + 0.5)**i
            x0 = cf.flatten()
            
            res = minimize(self._loss_function, x0, method='nelder-mead', options={'maxiter': 10, 'xatol': 50, 'adaptive': True})
            cf = res.x.reshape(1, self.ts_length, self.n_features)
            prob_target = self.model.predict(cf, verbose=0)[0][self.target_label]
            
        return cf[0]

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def _generate_cf_for_instance(args):
    """
    Worker function for multiprocessing.
    It loads its own model to avoid pickling errors and GPU conflicts.
    """
    # 1. Unpack arguments, now including instance_idx
    (instance_idx, instance, instance_label, model_path, 
     X_train_mad_scalar, ts_length, n_features, lambda_init) = args
    
    # 2. Add a logger context to track this specific instance in logs
    with logger.contextualize(instance_id=instance_idx):
        logger.debug(f"Worker started...") # DEBUG level won't show in console

        # 3. Configure TF to run on CPU
        import tensorflow as tf
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')
        from tensorflow import keras
        
        # 4. Load model
        try:
            model = keras.models.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None # Return None on failure

        # 5. Initialize generator and run task
        generator = WachterGenerator(
            model, X_train_mad_scalar, ts_length, n_features, lambda_init
        )
        
        try:
            cf = generator.generate(instance, instance_label)
            logger.debug(f"Worker finished successfully.")
            return cf
        except Exception as e:
            # Log the full exception
            logger.error(f"Generation failed: {e}")
            return None # Return None on failure

# --- UPDATED MAIN FUNCTION ---
def generate_wachter_cfs(dataset_name, data_dict, model_obj):
    """
    Main function to generate and save Wachter counterfactuals (PARALLELIZED).
    Note: 'model_obj' is passed from the dict but *not* used in workers.
    """
    logger.info(f'Generating Wachter counterfactuals for {dataset_name}...')
    
    # 1. Load data and model path
    X_train, y_train, X_test, y_test = data_dict[dataset_name]
    ts_length, n_features = X_train.shape[1], X_train.shape[2]
    
    # We pass the *path* to the workers, not the model object
    model_path = f'./models/{dataset_name}/{dataset_name}_best_model.hdf5'
    
    # 2. Get predictions (using the pre-loaded model object)
    logger.info("Generating initial predictions...")
    y_pred = np.argmax(model_obj.predict(X_test, verbose=0), axis=1)
    
    # 3. Calculate MAD scalar *once*
    X_train_mad_scalar = stats.median_abs_deviation(X_train.flatten())
    
    # 4. Create the list of tasks for the pool
    lambda_init = 0.1
    tasks = []
    for i in range(len(X_test)):
        task_args = (
            i, # <-- Pass the instance index for logging
            X_test[i], 
            y_pred[i], 
            model_path, 
            X_train_mad_scalar, 
            ts_length, 
            n_features, 
            lambda_init
        )
        tasks.append(task_args)

    # 5. Run the multiprocessing pool
    n_cores = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores free
    logger.info(f"Using {n_cores} cores for multiprocessing...")
    
    wcf_cfs_results = []
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        results_iterator = pool.imap(_generate_cf_for_instance, tasks)
        
        for cf in tqdm(results_iterator, total=len(tasks)):
            wcf_cfs_results.append(cf)
            
    logger.info("Multiprocessing complete.")

    # 6. Filter out failed runs (which returned None)
    wcf_cfs = [cf for cf in wcf_cfs_results if cf is not None]
    failed_count = len(wcf_cfs_results) - len(wcf_cfs)
    if failed_count > 0:
        logger.warning(f"Failed to generate {failed_count} counterfactuals. See DEBUG logs for details.")
    else:
        logger.info("All counterfactuals generated successfully.")

    # 7. Format and Store
    results = [{'cf': cf.reshape(1, ts_length, n_features), 'time': -1} for cf in wcf_cfs]
    
    save_path = f'./results/{dataset_name}/wcf_ng.pickle'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        
    logger.info(f"Saved Wachter counterfactuals for {dataset_name} to {save_path}")