import ast

import torch
from torch import nn, optim


class LambdaLayer(nn.Module):
    def __init__(self, lambda_expression):
        super(LambdaLayer, self).__init__()
        self.lambda_expression = lambda_expression
    def forward(self, x):
        return self.lambda_expression(x)

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        layer_specs: list,
        init_optimizer=False,
        learning_rate=0.01,
        optimizer=optim.Adam,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        '''
        :param input_size: the size of the input data
        :param layer_specs: a list of dictionaries, each containing "layer_type" + other key-value pairs, depending on the layer type:
            - "GRU" - "hidden_size", "num_layers"
            - "Linear" - "output_size"
            - "Lambda" - "lambda_expression"
                - "lambda_expression" must contain a python (lambda) function which takes a tensor as input and returns a tensor as output
                - examples (both are valid - string is converted to lambda expression):
                    "lambda_expression": lambda x: x * 1
                    "lambda_expression": "lambda x: x * 1"
                    "lambda_expression": """
                        def f(x):
                            return x * 1
                    """
            - Additional key-value pairs:
                - "activation" for the activation function which should be applied after the layer
        :param init_optimizer: if True, the optimizer is initialized with the given learning rate
        :param learning_rate: the learning rate for the optimizer
        :param optimizer: the optimizer to be used
        '''
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size

        for layer_spec in layer_specs:
            assert "layer_type" in layer_spec, "Each layer spec should contain a key 'layer_type'."
            
            ### layer initialization
            if layer_spec["layer_type"] == "GRU":
                assert "hidden_size" in layer_spec, "GRU layer spec should contain a key 'hidden_size'."
                assert "num_layers" in layer_spec, "GRU layer spec should contain a key 'num_layers'."
                self.layers.append(nn.GRU(
                    input_size=input_size,
                    hidden_size=layer_spec["hidden_size"],
                    num_layers=layer_spec["num_layers"],
                    batch_first=True,
                ))
                input_size = layer_spec["hidden_size"]
            elif layer_spec["layer_type"] == "Linear":
                assert "output_size" in layer_spec, "Linear layer spec should contain a key 'output_size'."
                self.layers.append(nn.Linear(
                    in_features=input_size,
                    out_features=layer_spec["output_size"],
                ))
                input_size = layer_spec["output_size"]
            elif layer_spec["layer_type"] == "Lambda":
                assert "lambda_expression" in layer_spec, "Lambda layer spec should contain a key 'lambda_expression'."
                lambda_expr = layer_spec["lambda_expression"]
                if type(lambda_expr) == str:
                    try:
                        # checks if the string contains a valid python code
                        ast.parse(lambda_expr)
                        lambda_expr = eval(lambda_expr)
                    except SyntaxError:
                        raise SyntaxError("Lambda expression is not valid.")
                self.layers.append(LambdaLayer(lambda_expr))
            else:
                raise ValueError("Unknown layer type: " + layer_spec["layer_type"])

            ### activation function
            if "activation" in layer_spec:
                self.layers.append(layer_spec["activation"])

        self.output_size = input_size
        self.device = device
        self.to(self.device)

        self.learning_rate = learning_rate if init_optimizer else None
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate) if init_optimizer else None

    def forward(self, input_data):
        for layer in self.layers:
            if type(layer) == nn.modules.rnn.GRU:
                input_data = layer(input_data)[0]
            else:
                input_data = layer(input_data)

        return input_data
