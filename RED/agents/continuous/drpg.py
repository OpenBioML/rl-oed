import copy

import numpy as np
import tensorflow as tf
from keras.api._v2 import keras


import logging

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras.utils import pad_sequences
from tensorflow import keras
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class ConcatLayer(nn.Module):
    def __init__(self, output_dim: int, concat_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.concat_dim = concat_dim

    def forward(self, x, y):
        return torch.cat([x, y], dim=self.concat_dim)


class GRUFCNetwork(nn.Module):
    def __init__(self, input_size: int, sequence_size: int, rec_sizes: list[int], hidden_sizes: list[int]):
        super().__init__()

        self.input_size = input_size
        self.sequence_size = sequence_size
        self.rec_sizes = rec_sizes
        self.hidden_sizes = hidden_sizes

        # GRU
        gru_layers = []
        prev_size = sequence_size
        for i, rec_size in enumerate(rec_sizes):
            layer = nn.GRU(input_size=prev_size, hidden_size=rec_size, batch_first=True)
            gru_layers.append(layer)
            prev_size = rec_size
        self.gru = nn.Sequential(*gru_layers)

        # Concat Layer
        concat_size = input_size + prev_size
        self.concat = ConcatLayer(concat_size)

        # FC layers
        lin_size = concat_size
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(lin_size, hidden_size))
            layers.append(nn.ReLU())
            lin_size = hidden_size

        self.fc = nn.Sequential(*layers)
        self.output_size = lin_size

    def forward(self, inputs):
        states, sequences = inputs

        gru_out = self.gru(sequences)
        concat_out = self.concat(states, gru_out)
        return self.fc(concat_out)


class CriticNetwork(nn.Module):
    """
    Network with GRUFCNetwork backbone, outputs scalar value
    """

    def __init__(self, input_size: int, sequence_size: int, rec_sizes: list[int], hidden_sizes: list[int]):
        super().__init__()

        self.input_size = input_size
        self.sequence_size = sequence_size
        self.rec_sizes = rec_sizes
        self.hidden_sizes = hidden_sizes

        self.backbone = GRUFCNetwork(input_size, sequence_size, rec_sizes, hidden_sizes)
        self.output = nn.Linear(self.backbone.output_size, 1)

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)
        return self.output(backbone_out)


class ActorNetwork(nn.Module):
    """
    Network with GRUFCNetwork backbone, outputs mean and log_std of the distribution over actions
    """

    def __init__(self, input_size: int, sequence_size: int, rec_sizes: list[int], hidden_sizes: list[int],
                 output_size: int):
        super().__init__()

        self.input_size = input_size
        self.sequence_size = sequence_size
        self.rec_sizes = rec_sizes
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.backbone = GRUFCNetwork(input_size, sequence_size, rec_sizes, hidden_sizes)
        self.mean = nn.Sequential(
            nn.Linear(self.backbone.output_size, output_size),
            nn.Sigmoid()
        )
        self.log_std = nn.Linear(self.backbone.output_size, output_size)

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)
        return self.mean(backbone_out), self.log_std(backbone_out)


class DRPGAgent:
    def __init__(self, layer_sizes: list[int], actor_network: nn.Module, critic_network: nn.Module = None,
                 learning_rate: float = 0.001):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.

        self.actor_network = actor_network
        self.critic_network = critic_network

        if critic_network is not None:
            self.critic_opt = torch.optim.Adam(self.critic_network.parameters(), lr=learning_rate)
            self.critic_loss = nn.MSELoss()

        self.actor_opt = torch.optim.Adam(self.actor_network.parameters(), lr=learning_rate)

        self.values = []
        self.actions = []

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequences = []
        self.next_sequences = []
        self.all_values = []

    def get_actions(self, inputs):

        states, sequences = inputs
        padded_seqs = pad_sequence(sequences, batch_first=True)
        mu, log_std = self.actor_network([states, padded_seqs])

        print('mu log_std', mu[0], log_std[0])

        actions = mu + torch.exp(log_std) * torch.randn_like(mu)
        return actions

    def loss(self, inputs, actions, returns):
        # Obtain mu and sigma from actor network
        mu, log_std = self.actor_network(inputs)

        # Compute log probability
        log_probability = self.log_probability(actions, mu, log_std)
        print('log probability', log_probability.shape)
        print('returns:', returns.shape)

        # Compute weighted loss
        loss_actor = - torch.mean(torch.multiply(returns, log_probability))
        print('loss actor', loss_actor.shape)
        return loss_actor

    def log_probability(self, actions, mu, log_std, eps: float = 1e-8):
        pre_sum = -0.5 * (((actions - mu) / (torch.exp(log_std) + eps)) ** 2 + 2 * log_std + np.log(2 * torch.pi))
        return torch.sum(pre_sum, dim=1)

    def Q_update(self, policy: bool, *args, **kwargs):
        inputs, actions, returns = self.get_inputs_targets()
        if self.critic_network is not None:
            expected_returns: torch.Tensor = self.critic_network(inputs)
            returns -= expected_returns.reshape(-1)
            self.critic_opt.zero_grad()
            outputs = self.critic_network(inputs)
            loss = self.critic_loss(outputs, returns)
            loss.backward()
            self.critic_opt.step()

        if policy:
            self.actor_network.zero_grad()
            loss = self.loss(inputs, actions, returns)
            loss.backward()
            self.actor_opt.step()

    def get_inputs_targets(self):
        '''
        Gets fitted Q inputs and calculates targets for training the Q-network for episodic training.
        :return: inputs, actions, all_values
        '''

        # iterate over all experience in memory and create fitted Q targets
        for i, trajectory in enumerate(self.memory):
            e_rewards = []
            sequence = [[0] * self.layer_sizes[1]]
            for j, transition in enumerate(trajectory):
                self.sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done, u = transition
                sequence.append(np.concatenate((state, u / 1)))
                # one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10
                self.next_sequences.append(copy.deepcopy(sequence))
                self.states.append(state)
                self.next_states.append(next_state)
                self.actions.append(action)
                self.rewards.append(reward)
                e_rewards.append(reward)
                self.dones.append(done)

            e_values = [e_rewards[-1]]

            for i in range(2, len(e_rewards) + 1):
                e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
            self.all_values.extend(e_values)

        padded = pad_sequences(self.sequences, maxlen=11, dtype='float64')
        states = np.array(self.states)
        actions = np.array(self.actions)
        all_values = np.array(self.all_values)

        self.sequences = []
        self.states = []
        self.actions = []
        self.all_values = []
        self.memory = []  # reset memory after this information has been extracted

        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]
        actions = actions[randomize]

        padded = padded[randomize]
        all_values = all_values[randomize]

        inputs = [states, padded]
        print('inputs, actions, all_values', inputs[0].shape, inputs[1].shape, actions.shape, all_values.shape)
        return inputs, actions, all_values

    def save_network(self, save_path):  # tested
        # print(self.network.layers[1].get_weights())
        self.actor_network.save(save_path + '/saved_network.h5')

    def load_network(self, load_path):  # tested
        try:
            self.actor_network = keras.models.load_model(
                load_path + '/saved_network.h5')  # sometimes this crashes, apparently a bug in keras

        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.actor_network.load_weights(
                load_path + '/saved_network.h5')  # this requires model to be initialised exactly the same


def get_drpg_agent(layer_sizes, critic: bool, learning_rate: float):
    input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes
    actor_network = ActorNetwork(input_size, sequence_size, rec_sizes, hidden_sizes, output_size)
    if critic:
        critic_network = CriticNetwork(input_size, sequence_size, rec_sizes, hidden_sizes)
    else:
        critic_network = None
    return DRPGAgent(layer_sizes, actor_network, critic_network, learning_rate)
