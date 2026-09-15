import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        h = initial_hidden_state
        c = initial_cell_state
       
        outputs = []

        for vals in x:
            vals = vals.reshape(self.input_size, 1)
            combined = torch.cat((h, vals), dim = 0)

            f = torch.sigmoid(self.Wf @ combined + self.bf)

            i = torch.sigmoid(self.Wi @ combined + self.bi)

            c_tilde = torch.tanh(self.Wc @ combined + self.bc)

            c = f * c + i * c_tilde

            o = torch.sigmoid(self.Wo @ combined + self.bo)

            h = o * torch.tanh(c)

            outputs.append(h)

        outputs = torch.stack(outputs, dim=0)

        return outputs, h, c
