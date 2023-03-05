import torch
from torch import nn, optim
from torch.nn.modules import Module

class LambdaLayer(nn.Module):
    def __init__(self, lambda_expression):
        super(LambdaLayer, self).__init__()
        self.lambda_expression = lambda_expression
    def forward(self, x):
        return self.lambda_expression(x)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layers_data: list, learning_rate=0.01, optimizer=optim.Adam):
        '''
          input_size:
          layers_data: 4-element tuple list containing: the type of layer to be created, the layer size, the activation functions and an optional lambda expression (for lambda layers)
          learning_rate:
          optimizer:
        '''
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size

        for layer_id in range(len(layers_data)):
          layer_type, sizes, activation, lambda_expression = layers_data[layer_id]

          if layer_type == 'GRU':
            hidden_size, num_layers = sizes
            self.layers.append(nn.GRU(input_size, hidden_size, num_layers))
            input_size = hidden_size

          elif layer_type == 'Linear':
            output_size = sizes
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

          elif layer_type == 'Lambda':
            self.layers.append(LambdaLayer(lambda_expression))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    def forward(self, input_data):
        for layer in self.layers:
            if type(layer) == nn.modules.rnn.GRU:
              input_data = layer(input_data)[0]
            else:
              input_data = layer(input_data)

        return input_data
