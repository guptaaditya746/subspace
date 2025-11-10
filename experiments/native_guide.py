#!/usr/bin/env python
# coding: utf-8

# Contains all logic for the Native Guide counterfactual method.

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tslearn.neighbors import KNeighborsTimeSeries
import tensorflow as tf

def native_guide_retrieval(query, predicted_label, distance, n_neighbors, X_train, y_train):
    """
    Finds the Nearest Unlike Neighbor (NUN).
    """
    df = pd.DataFrame(y_train, columns = ['label'])
    df.index.name = 'index'
    # Get indices for all classes *not* equal to the predicted label
    unlike_indices = df[df['label'] != predicted_label].index.values
    ts_length = X_train.shape[1]

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
    knn.fit(X_train[unlike_indices])
    
    dist, ind = knn.kneighbors(query.reshape(1, ts_length), return_distance=True)

    # Return the distance and the *original* index from X_train
    original_index = unlike_indices[ind[0][:]]
    return dist[0], original_index


def findSubarray(a, k):
    """
    Used to find the maximum contiguous subarray of length k in the explanation weight vector.
    """
    n = len(a)
    vec=[] 

    # Iterate to find all the sub-arrays 
    for i in range(n-k+1): 
        temp=[] 
        # Store the sub-array elements in the array 
        for j in range(i,i+k): 
            temp.append(a[j]) 
        # Push the vector in the container 
        vec.append(temp) 

    sum_arr = [np.sum(v) for v in vec]
    return (vec[np.argmax(sum_arr)])


def counterfactual_generator_swap(instance_data, nun_data, instance_label, cam_weights_nun, model, min_subarray_length=1):
    """
    Generates the counterfactual by swapping in the most influential subarray from the NUN.
    
    instance_data: The X_test instance (2D array)
    nun_data: The X_train NUN instance (2D array)
    instance_label: The predicted label (int) for the instance
    cam_weights_nun: The 1D array of CAM weights for the NUN
    model: The trained Keras model
    """
    
    subarray_length = min_subarray_length
    ts_length = len(instance_data)
    
    # Initial setup
    most_influencial_array = findSubarray(cam_weights_nun, subarray_length)
    # Find first occurrence of the start of the subarray
    starting_point = np.where(cam_weights_nun == most_influencial_array[0])[0][0]
    
    # Ensure subarray does not go out of bounds
    end_point = min(starting_point + subarray_length, ts_length)
    
    X_example = np.concatenate((
        instance_data[:starting_point], 
        nun_data[starting_point:end_point], 
        instance_data[end_point:]
    ))
    
    # Reshape for model prediction
    X_example_reshaped = X_example.reshape(1, -1, 1)
    prob_target = model.predict(X_example_reshaped, verbose=0)[0][instance_label]

    # Keep expanding the subarray until the prediction flips
    while prob_target > 0.5:
        subarray_length += 1
        if subarray_length > ts_length:
            # Failsafe: if the whole series is swapped and still no flip
            print(f"Warning: Could not flip label even after full swap.")
            break 
            
        most_influencial_array = findSubarray(cam_weights_nun, subarray_length)
        starting_point = np.where(cam_weights_nun == most_influencial_array[0])[0][0]
        end_point = min(starting_point + subarray_length, ts_length)

        X_example = np.concatenate((
            instance_data[:starting_point], 
            nun_data[starting_point:end_point], 
            instance_data[end_point:]
        ))
        
        X_example_reshaped = X_example.reshape(1, -1, 1)
        prob_target = model.predict(X_example_reshaped, verbose=0)[0][instance_label]

    return X_example


def generate_native_guide_cfs(dataset_name, data_dict, models_dict):
    """
    Main function to generate and save Native Guide counterfactuals for a dataset.
    """
    print(f'Generating Native Guide counterfactuals for {dataset_name}...')
    
    # Load data and model
    X_train, y_train, X_test, y_test = data_dict[dataset_name]
    model = models_dict[dataset_name]
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # 1. Get the NUNs
    nuns_idx = []
    for instance_idx in range(len(X_test)):
        # Find the single nearest unlike neighbor (k=1)
        nun_index = native_guide_retrieval(X_test[instance_idx], y_pred[instance_idx], 'euclidean', 1, X_train, y_train)[1][0]
        nuns_idx.append(nun_index)
    nuns_idx = np.array(nuns_idx)

    # 2. Get cam importances
    try:
        cam_training_weights = np.load(f'../methods/NativeGuide/Class_Activation_Mapping/{dataset_name}_cam_train_weights.npy')
    except FileNotFoundError:
        print(f"Error: Could not find CAM weights at ./methods/NativeGuide/Class_Activation_Mapping/{dataset_name}_cam_train_weights.npy")
        return

    # 3. Get the counterfactuals 
    ng_cfs = []
    test_instances_idx = np.array(range(len(X_test)))
    
    for test_instance_idx, nun_idx in tqdm(zip(test_instances_idx, nuns_idx), total=len(test_instances_idx)):
        
        instance_data = X_test[test_instance_idx]
        nun_data = X_train[nun_idx]
        instance_label = y_pred[test_instance_idx]
        cam_weights_nun = cam_training_weights[nun_idx]
        
        cf = counterfactual_generator_swap(
            instance_data, nun_data, instance_label, cam_weights_nun, model, min_subarray_length=1
        )
        ng_cfs.append(cf)

    # 4. Store
    # Adapt counterfactual result to standard format
    results = [{'cf': np.expand_dims(cf, axis=0), 'time': -1} for cf in ng_cfs]
    
    save_path = f'./experiments/results/{dataset_name}/ng.pickle'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        
    print(f"Saved Native Guide counterfactuals for {dataset_name} to {save_path}")
