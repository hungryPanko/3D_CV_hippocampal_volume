"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/Users/hannalee/Documents/Udacity/AI_healcare/3D_FinalProj/section1/out"
        self.n_epochs = 5
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/Users/hannalee/Documents/Udacity/AI_healcare/3D_FinalProj/section2/result"

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

    # TASK: create three keys in the dictionary: "train", "val" and "test".
    train_size = int(0.7 * len(keys))
    val_size = int(0.2 * len(keys))
    print("\t    train_size: ", train_size)
    print("\t    val_size: ", val_size)

    split["train"] = keys[:train_size]
    split["val"] = keys[train_size:train_size + val_size]
    split["test"] = keys[train_size + val_size:]

    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    print("\nStart Running")
    exp.run()

    # prep and run testing
    # TASK: Test method is not complete. Go to the method and complete it
    print("\nStart Testing")
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

#lsof -i:6006
# kill -9 PID