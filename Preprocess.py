import os

"""
The preprocess class will save a copy of the target files in the writingPrompts dataset
These target files will be what the model learns of 
On the Kaggle page for the dataset they recommend truncating the stories to 1000 words so that will be implemented here and expermented with
"""

class Preprocess:

    def __init__(self, dataset_path, max_story_length=1000, save_name="_processed"):
        self.dataset_path = dataset_path
        self.max_story_length = max_story_length
        self.save_name = save_name

    # Truncate stories to max_story_length and save to new files
    def truncate_stories(self):
        data = ['test', 'train', 'valid']
        for name in data:

            input_path = os.path.join(self.dataset_path, f"{name}.wp_target")
            output_path = os.path.join(self.dataset_path, f"{name}{self.save_name}.wp_target")

            # Check if input file exists
            if not os.path.exists(input_path):
              print(f"File {input_path} does not exist. Skipping!")
              continue
            
            # Open file and read stories (There is one story per line)
            with open(input_path,  encoding="utf-8") as f:
                stories = f.readlines()

            truncated_stories = [" ".join(story.split()[:self.max_story_length]) for story in stories]

            with open(output_path, "w", encoding="utf-8") as f:
              for story in truncated_stories:
                f.write(story.strip() + "\n")

if __name__ == "__main__":
    
    dataset_path = "./writingPrompts"
    preprocessor = Preprocess(dataset_path)
    preprocessor.truncate_stories()
