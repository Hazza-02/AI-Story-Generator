"""
Thought process:

The model recieves a text input from the source files that relates to various prompts for stories.
For instance, a prompt could be a about making a story about "a Knight in a land of dragons"
The model will learn to generate stories based on these prompts by training on the target files in the writingPrompts dataset.
The target files contain the actual stories that correspond to the prompts.

I am thinking of using a Long Short-Term Memory (LSTM) network for this project but can experiment with others aswell.
From what I have seen standard RNNs struggle to rember the original context of when generating text due to vanishing gradients,
so LSTMs are a natural choice to solve this issue.

Other ideas:
- Transformers

"""