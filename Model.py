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

import torch
import torch.nn as nn

# Basic LSTM model from "https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/"
# Will need to modify this for optimal results, not sure how good it will be for story generation
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, hn, cn
