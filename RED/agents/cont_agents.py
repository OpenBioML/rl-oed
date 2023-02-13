import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import numpy as np

class DRPG_agent(nn.Module):
    def __init__(self, layer_sizes, learning_rate=0.001, critic=True):
        super(DRPG_agent, self).__init__()
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.

        self.critic = critic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if critic:
            self.critic_network = self.initialise_network(layer_sizes, critic_nw=True)
            self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
        self.actor_network = self.initialise_network(layer_sizes)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)

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

    def initialise_network(self, layer_sizes, critic_nw=False):
        input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes
        layers = []

        self.sequence_input_size = (None, sequence_size)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        for i, rec_size in enumerate(rec_sizes):
            layers.append(nn.GRU(input_size=sequence_size, hidden_size=rec_size, batch_first=True))

        layers.append(nn.Linear(input_size + rec_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        for i, hl_size in enumerate(hidden_sizes[1:]):
            layers.append(nn.Linear(hidden_sizes[i], hl_size))
            layers.append(nn.ReLU())

        if critic_nw:
            layers.append(nn.Linear(hidden_sizes[-1], 1))
        else:
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_sizes[-1], output_size))

        return nn.Sequential(*layers)

def get_actions(self, inputs):
    states, sequences = inputs

    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)

    mu, log_std = self.actor_network(states, sequences)

    actions = mu + torch.randn_like(mu) * torch.exp(log_std)

    return actions

def loss(self, inputs, actions, returns):
    mu, log_std = self.actor_network(inputs)

    log_probability = self.log_probability(actions, mu, log_std)

    loss_actor = -torch.mean(returns * log_probability)

    return loss_actor

def log_probability(self, actions, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (((actions - mu) / (torch.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))

    return torch.sum(pre_sum, dim=1)

def policy_update(self):
    inputs, actions, returns = self.get_inputs_targets()

    if self.critic:
        expected_returns = self.critic_network(inputs)

        returns -= expected_returns.reshape(-1)

        self.critic_network.fit(inputs, returns)

    loss = self.loss(inputs, actions, returns)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

def get_inputs_targets(self):
    # Calculate targets for training the Q-network for episodic training
    sequences = []
    states = []
    actions = []
    rewards = []
    all_values = []

    # Iterate over all experiences in memory and create fitted Q targets
    for i, trajectory in enumerate(self.memory):
        e_rewards = []
        sequence = [[0] * self.layer_sizes[1]]
        for j, transition in enumerate(trajectory):
            sequences.append(copy.deepcopy(sequence))
            state, action, reward, _, done, u = transition
            sequence.append(np.concatenate((state, u / 1)))
            rewards.append(reward)
            e_rewards.append(reward)

        e_values = [e_rewards[-1]]
        for i in range(2, len(e_rewards) + 1):
            e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
        all_values.extend(e_values)
        states.extend([transition[0] for transition in trajectory])
        actions.extend([transition[1] for transition in trajectory])

    padded = pad_sequences(sequences, maxlen=11, dtype='float64')
    states = torch.tensor(states)
    actions = torch.tensor(actions)
    all_values = torch.tensor(all_values)

    self.memory = []  # reset memory after this information has been extracted

    randomize = torch.randperm(len(states))

    states = states[randomize]
    actions = actions[randomize]
    padded = padded[randomize]
    all_values = all_values[randomize]

    inputs = (states, padded)
    print('inputs, actions, all_values', inputs[0].shape, inputs[1].shape, actions.shape, all_values.shape)
    return inputs, actions, all_values

def save_network(self, save_path):
    torch.save(self.network.state_dict(), save_path + '/saved_network.pt')

def load_network(self, load_path):
    try:
        self.network.load_state_dict(torch.load(load_path + '/saved_network.pt'))
    except:
        print('EXCEPTION IN LOAD NETWORK')
        raise Exception("Loading network failed.")




