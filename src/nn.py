from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NetContext:
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function
