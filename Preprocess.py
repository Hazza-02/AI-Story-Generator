import os

"""
The preprocess class will save a copy of the target files in the writingPrompts dataset
These target files will be what the model learns of 
On the Kaggle page for the dataset they recommend truncating the stories to 1000 words so that will be implemented here and expermented with
"""

class Preprocess:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
