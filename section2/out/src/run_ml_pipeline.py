"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"D:/CodeBase/AI-Healthcare-Projects/MRIHippocampalVolume/nd320-c3-3d-imaging-starter/section1/out/data/TrainingSet/"
        self.n_epochs = 15
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.patch_size = 64
        self.test_results_dir = "D:/CodeBase/AI-Healthcare-Projects/MRIHippocampalVolume/nd320-c3-3d-imaging-starter/section2/out/results/"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>

    # Define dataset length (assuming 'data' is a list or array)
    num_samples = len(data)

    # Randomly shuffle indices
    shuffled_keys = np.random.RandomState(seed=55).permutation(keys)

    # Define split ratios (e.g., 70% train, 20% val, 10% test)
    train_ratio, val_ratio = 0.7, 0.2
    train_split = int(train_ratio * num_samples)
    val_split = int(val_ratio * num_samples)

    # Assign splits
    split = {
        "train": shuffled_keys[:train_split],              # First 70% for training
        "val": shuffled_keys[train_split:train_split + val_split],  # Next 20% for validation
        "test": shuffled_keys[train_split + val_split:]    # Remaining 10% for testing
    }

    # Print results
    print(f"Train set size: {len(split['train'])}")
    print(f"Validation set size: {len(split['val'])}")
    print(f"Test set size: {len(split['test'])}")

    # # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

