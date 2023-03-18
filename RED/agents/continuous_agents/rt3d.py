import copy
import gc
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

from RED.utils.network import NeuralNetwork


class RT3D_agent():
    '''
    Class that implements the Recurrent Twin Delayed Deep Deterministic Policy Gradient agent (RT3D).
    '''
    def __init__(
        self,
        val_module_specs,
        pol_module_specs,
        val_learning_rate=0.001,
        pol_learning_rate=0.001,
        batch_size=256,
        action_bounds=[0, 1],
        noise_bounds=[-0.25, 0.25],
        noise_std=0.1,
        gamma=1,
        polyak=0.995,
        mem_size=500_000_000,
        max_length=11,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        '''
        Initialises the RT3D agent.
        :param val_module_specs: module specifications for the Q networks
            - last module gets the concatenated output of all the previous modules
            - list of dictionaries, each dictionary containing keys "input_size" and "layers"
                - "input_size" is the input size of the module
                - "layers" is a list of dictionaries, each containing "layer_type" and other key-value pairs, depending on the layer type:
                    - "GRU" - "hidden_size", "num_layers"
                    - "Linear" - "output_size"
                    - "Lambda" - "lambda_expression"
                    - Additional key-value pairs:
                        - "activation" for the activation function which should be applied after the layer
        :param pol_module_specs: module specifications for the policy networks
            - last module gets the concatenated output of all the previous modules
            - list of dictionaries, each dictionary containing keys "input_size" and "layers"
                - "input_size" is the input size of the module
                - "layers" is a list of dictionaries, each containing "layer_type" and other key-value pairs, depending on the layer type:
                    - "GRU" - "hidden_size", "num_layers"
                    - "Linear" - "output_size"
                    - "Lambda" - "lambda_expression"
                    - Additional key-value pairs:
                        - "activation" for the activation function which should be applied after the layer
        :param val_learning_rate: learning rate for the value networks
        :param pol_learning_rate: learning rate for the policy network
        :param batch_size: batch size for training the networks
        :param action_bounds: bounds for the actions
        :param noise_bounds: bounds for the noise added to the actions
        :param noise_std: standard deviation of the noise added to the actions
        :param gamma: discount rate
        :param polyak: polyak averaging rate (see method update_target_networks)
        :param mem_size: size of the replay buffer
        :param max_length: max sequence length
        :param device: device to use for pytorch operations
        '''
        self.val_module_specs = val_module_specs
        self.pol_module_specs = pol_module_specs
        self.val_learning_rate = val_learning_rate
        self.pol_learning_rate = pol_learning_rate
        self.batch_size = batch_size
        self.action_bounds= action_bounds
        self.noise_bounds = noise_bounds
        self.noise_std = noise_std
        self.gamma = gamma
        self.polyak = polyak
        self.mem_size = mem_size
        self.max_length = max_length
        self.device = device

        ### initialise policy networks
        # policy network (base)
        self.policy_network = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in pol_module_specs
        ])
        self.seq_size = self.policy_network[0].input_size
        self.policy_out_size = self.policy_network[-1].output_size
        self.policy_network_opt = Adam(self.policy_network.parameters(), lr=self.pol_learning_rate)
        self.policy_network_loss = lambda predicted_action_values: -torch.mean(predicted_action_values)
        # policy network (target)
        self.policy_target = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in pol_module_specs
        ])

        ### initialise value networks
        # Q-value network 1 (base)
        self.Q1_network = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in val_module_specs
        ])
        self.Q1_network_opt = Adam(self.Q1_network.parameters(), lr=val_learning_rate)
        self.Q1_network_loss = nn.MSELoss()
        # Q-value network 1 (target)
        self.Q1_target = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in val_module_specs
        ])

        # Q-value network 2 (base)
        self.Q2_network = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in val_module_specs
        ])
        self.Q2_network_opt = Adam(self.Q2_network.parameters(), lr=val_learning_rate)
        self.Q2_network_loss = nn.MSELoss()
        # Q-value network 2 (target)
        self.Q2_target = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in val_module_specs
        ])

        ### initialise buffers
        self.memory = []
        self.values = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequences = []
        self.next_sequences = []
        self.all_returns = []

    def initialise_network(self, input_size, layer_specs, init_optimizer=False):
        '''
        Initialises a neural network.
        :param input_size: input size of the network
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
        :return: initialised neural network
        '''

        network = NeuralNetwork(
            input_size=input_size,
            layer_specs=layer_specs,
            init_optimizer=init_optimizer,
            device=self.device,
        )
        return network

    def get_actions_dist(self, inputs, explore_rate, test_episode=False, recurrent=True):
        '''
        Gets actions by adding random noise to the actions.
        :param inputs: inputs for policy network
        :param explore_rate: probability of taking a random action
        :param test_episode: whether the episode is a test episode
        :param recurrent: whether to use recurrent network
        :return: actions
        '''

        actions = self.forward_policy_net(
            policy_net=self.policy_network,
            inputs=inputs,
            recurrent=recurrent
        )

        actions = actions.detach().cpu().numpy()

        if test_episode:
            actions[:-1] += np.random.normal(0, explore_rate, size=actions[:-1].shape)
        else:
            actions += np.random.normal(0, explore_rate, size=actions.shape)
        
        actions = np.clip(actions, self.action_bounds[0], self.action_bounds[1])

        return actions

    def get_actions(self, inputs, explore_rate, test_episode=False, recurrent=True, return_exploit_flags=False):
        '''
        Gets a mix of explore/exploit actions between the min and max bounds (exploration with probablilty explore_rate uniformly distributed).
        :param inputs: inputs for policy network
        :param explore_rate: probability of taking a random action
        :param test_episode: whether the episode is a test episode
        :param recurrent: whether to use recurrent network
        :return: actions
        '''
        
        if recurrent:
            states, sequences = inputs # states [batch, features], sequences [batch, sequence, features]
        else:
            states = inputs[0]

        ### choose between explore/exploit
        if test_episode:
            rng = np.random.random(len(states) - 1)
        else:
            rng = np.random.random(len(states))

        explore_inds = np.where(rng < explore_rate)[0]
        exploit_inds = np.where(rng >= explore_rate)[0]

        if test_episode:
            exploit_inds = np.append(exploit_inds, len(states) - 1)

        if return_exploit_flags:
            exploit_flags = np.zeros((len(states)), dtype=np.int32)
            exploit_flags[exploit_inds] = 1
        
        ### get actions
        actions = torch.zeros((len(states), self.policy_out_size), dtype=torch.float32, device=self.device)

        # explore actions (uniformly distributed between the action bounds)
        explore_actions = (self.action_bounds[1] - self.action_bounds[0]) \
            * torch.rand((len(explore_inds), self.policy_out_size), dtype=torch.float32, device=self.device) \
            + self.action_bounds[0]
        actions[explore_inds] = explore_actions

        # exploit actions (policy network)
        if len(exploit_inds) > 0:
            # prepare inputs
            if recurrent:
                policy_net_inputs = [np.array(states)[exploit_inds], np.array(sequences)[exploit_inds]]
            else:
                policy_net_inputs = [np.array(states)[exploit_inds]]
            # run through the policy network
            exploit_actions = self.forward_policy_net(policy_net=self.policy_network, inputs=policy_net_inputs, recurrent=recurrent)
            # add noise
            exploit_actions += torch.normal(0, explore_rate * self.noise_std * 2, size=exploit_actions.shape, device=self.device)
            actions[exploit_inds] = exploit_actions

        # clip
        actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
        
        actions = actions.cpu().detach().numpy()

        return actions if not return_exploit_flags else (actions, exploit_flags)

    def forward_policy_net(self, policy_net: nn.Module, inputs, recurrent=True):
        '''
        Forward pass of the policy net given inputs.
        :param inputs: inputs for policy network
        :param recurrent: whether to use the recurrent network
        :return: outputs of the given policy net
        '''

        if recurrent:
            ### prepare inputs for pytorch's models
            states, sequences = inputs
            # states [batch, features]
            if type(states) != torch.Tensor:
                states = torch.tensor(states, dtype=torch.float32, device=self.device)
            # sequences [batch, sequence, features]
            if type(sequences) in (list, tuple):
                sequences = pad_sequence(
                    [torch.tensor(seq[-self.max_length:], dtype=torch.float32) for seq in sequences],
                    batch_first=True
                ).to(self.device)
            elif type(sequences) == np.ndarray:
                sequences = torch.tensor(sequences[:,-self.max_length:], dtype=torch.float32, device=self.device)
            elif type(sequences) != torch.Tensor:
                raise ValueError("Sequences must be a list, tuple, or numpy array")
            
            ### run through the policy network
            recurrent_out = policy_net[0](sequences)[:,-1,:] # last output of the sequence
            head_inps = torch.cat((states, recurrent_out), dim=1)
            policy_net_out = policy_net[1](head_inps)
        else:
            head_inps = torch.tensor(inputs[0], dtype=torch.float32, device=self.device)
        
        policy_net_out = policy_net[1](head_inps)
        
        return policy_net_out

    def forward_q_net(self, q_net: nn.Module, inputs, recurrent=True):
        '''
        Forward pass of the given Q net with the inputs.
        :param inputs: inputs for Q network
        :param recurrent: whether to use the recurrent network
        :return: outputs of the given Q net
        '''

        if recurrent:
            ### prepare inputs for pytorch's models
            state_actions, sequences = inputs
            # state-action pairs [batch, features]
            if type(state_actions) != torch.Tensor:
                state_actions = torch.tensor(state_actions, dtype=torch.float32, device=self.device)
            # sequences [batch, sequence, features]
            if type(sequences) in (list, tuple):
                sequences = pad_sequence(
                    [torch.tensor(seq[-self.max_length:], dtype=torch.float32) for seq in sequences],
                    batch_first=True
                ).to(self.device)
            elif type(sequences) == np.ndarray:
                sequences = torch.tensor(sequences[:,-self.max_length:], dtype=torch.float32, device=self.device)
            elif type(sequences) != torch.Tensor:
                raise ValueError("Sequences must be a list, tuple, or numpy array")
            
            ### run through the Q network
            recurrent_out = q_net[0](sequences)[:,-1,:] # last output of the sequence
            head_inps = torch.cat((state_actions, recurrent_out), dim=1)
        else:
            state_actions = inputs[0]
            head_inps = torch.tensor(state_actions, dtype=torch.float32, device=self.device)
        
        q_net_out = q_net[1](head_inps)

        return q_net_out

    def get_inputs_targets(self, recurrent=True, monte_carlo=False):
        '''
        --- Use get_inputs_targets_low_mem() instead, this is *not* more efficient ---
        --- TODO: fix or remove ---
        assembles the Q learning inputs and trgets from agents memory
        :param recurrent:
        :param monte_carlo:
        :return:
        '''

        ### collect the data
        sample_size = int(self.batch_size * 10)
        sample_idxs = np.random.randint(0, min(self.mem_size, len(self.memory)), size=(sample_size))
        for i, trajectory in enumerate(self.memory):
            e_rewards = []
            sequence = [[0]*self.seq_size]
            for j, transition in enumerate(trajectory):
                state, action, reward, next_state, done = transition

                self.sequences.append(copy.deepcopy(sequence))
                sequence.append(np.concatenate((state, action)))
                self.next_sequences.append(copy.deepcopy(sequence))
                self.states.append(state)
                self.next_states.append(next_state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                e_rewards.append(reward)

            if monte_carlo:
                e_values = [e_rewards[-1]]

                for i in range(2, len(e_rewards) + 1):
                    e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
                self.all_returns.extend(e_values)

            # remove items if agents memory is full
            if len(self.states) > self.mem_size:
                del self.sequences[:len(self.states)-self.mem_size]
                del self.next_sequences[:len(self.states)-self.mem_size]
                del self.next_states[:len(self.states)-self.mem_size]
                del self.actions[:len(self.states)-self.mem_size]
                del self.rewards[:len(self.states)-self.mem_size]
                del self.dones[:len(self.states)-self.mem_size]
                del self.states[:len(self.states) - self.mem_size]

        # TODO: this is really memory inefficient, take random sample before initialising arrays
        next_states = np.array(self.next_states, dtype=np.float64)[:self.mem_size]
        rewards = np.array(self.rewards).reshape(-1, 1)[:self.mem_size]
        dones = np.array(self.dones).reshape(-1, 1)[:self.mem_size]
        states = np.array(self.states)[:self.mem_size]
        actions = np.array(self.actions)[:self.mem_size]
        all_returns = np.array(self.all_returns)[:self.mem_size]
        sequences = self.sequences[:self.mem_size]
        next_sequences = self.next_sequences[:self.mem_size]

        self.memory = self.memory[-self.mem_size:]  # reset memory after this information has been extracted

        if monte_carlo : # only take last experiences
            '''
            batch_size = self.batch_size
            if states.shape[0] > batch_size:
                states = states[-batch_size:]
                padded = padded[-batch_size:]
                next_padded = next_padded[-batch_size:]
                next_states = next_states[-batch_size:]
                actions = actions[-batch_size:]
                rewards = rewards[-batch_size:]
                dones = dones[-batch_size:]
                all_returns = all_returns[-batch_size:]
            '''
            targets = all_returns
            pass
        else:
            ### take random sample
            sample_size = int(self.batch_size * 10)
            sample_idxs = np.random.randint(max(0, states.shape[0] - self.mem_size), states.shape[0], size=(sample_size))

            states = states[sample_idxs]
            next_states = next_states[sample_idxs]
            actions = actions[sample_idxs]
            rewards = rewards[sample_idxs]
            dones = dones[sample_idxs]
            sequences = [sequences[i] for i in sample_idxs]
            next_sequences = [next_sequences[i] for i in sample_idxs]

            ### get next actions from target policy
            next_actions = self.forward_policy_net(
                policy_net=self.policy_target,
                inputs=[next_states, next_sequences],
                recurrent=recurrent,
            ).cpu().detach().numpy()

            # target policy smoothing
            noise = np.clip(np.random.normal( 0, self.noise_std, next_actions.shape), self.noise_bounds[0], self.noise_bounds[1])
            next_actions = np.clip(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

            ### get next values from target Q networks
            self.Q1_target.eval()
            with torch.no_grad():
                Q1 = self.forward_q_net(
                    q_net=self.Q1_target,
                    inputs=[np.concatenate((next_states, next_actions), axis=1), next_sequences],
                    recurrent=recurrent
                ).cpu().detach().numpy()
            
            self.Q2_target.eval()
            with torch.no_grad():
                Q2 = self.forward_q_net(
                    q_net=self.Q2_target,
                    inputs=[np.concatenate((next_states, next_actions), axis=1), next_sequences],
                    recurrent=recurrent
                ).cpu().detach().numpy()

            next_values = np.minimum(Q1, Q2)
            targets = rewards + self.gamma * (1 - dones) * next_values

        ### shuffle the data and construct the inputs and targets
        randomize = np.arange(len(states))
        np.random.shuffle(randomize)
        states = states[randomize]
        actions = actions[randomize]
        sequences = [sequences[i] for i in randomize]
        targets = targets[randomize]
        inputs = [states, sequences]
        targets = targets[randomize]

        gc.collect() # clear old stuff from memory
        return inputs, actions, targets

    def get_inputs_targets_low_mem(self, recurrent=True, monte_carlo=False):
        '''
        Assembles the Q learning inputs and targets from agent's memory, uses less memory but is slower.
        :param recurrent: whether to use the recurrent networks
        :param monte_carlo:
        :return: inputs, actions, targets
        '''

        # TODO: enable all the options here
        self.memory = self.memory[-self.mem_size:]

        sequences = []
        next_sequences = []
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        ### collect the data
        sample_size = int(self.batch_size * 10)
        sample_idxs = np.random.randint(0, min(self.mem_size, len(self.memory)), size=(sample_size))
        for trajectory_idx in sample_idxs:
            trajectory = self.memory[trajectory_idx]
            sequence = [np.array([0] * self.seq_size)]
            for transition in trajectory:
                state, action, reward, next_state, done = transition

                sequences.append(copy.deepcopy(np.array(sequence)))
                sequence.append(np.concatenate((state, action)))
                next_sequences.append(copy.deepcopy(np.array(sequence)))
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

        next_states = np.array(next_states, dtype=np.float64)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        states = np.array(states)
        actions = np.array(actions)

        ### get next actions from target policy
        next_actions = self.forward_policy_net(
            policy_net=self.policy_target,
            inputs=[next_states, next_sequences],
            recurrent=recurrent,
        ).cpu().detach().numpy()

        # target policy smoothing
        noise = np.clip(np.random.normal(0, self.noise_std, next_actions.shape), self.noise_bounds[0], self.noise_bounds[1])
        next_actions = np.clip(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

        ### get next values from target Q networks
        self.Q1_target.eval()
        with torch.no_grad():
            Q1 = self.forward_q_net(
                q_net=self.Q1_target,
                inputs=[np.concatenate((next_states, next_actions), axis=1), next_sequences],
                recurrent=recurrent
            ).cpu().detach().numpy()
        
        self.Q2_target.eval()
        with torch.no_grad():
            Q2 = self.forward_q_net(
                q_net=self.Q2_target,
                inputs=[np.concatenate((next_states, next_actions), axis=1), next_sequences],
                recurrent=recurrent
            ).cpu().detach().numpy()

        next_values = np.minimum(Q1, Q2)
        targets = rewards + self.gamma * (1 - dones) * next_values

        ### shuffle the data and construct the inputs and targets
        randomize = np.arange(len(states))
        np.random.shuffle(randomize)
        states = states[randomize]
        actions = actions[randomize]
        sequences = [sequences[i] for i in randomize]
        targets = targets[randomize]
        inputs = [states, sequences]

        return inputs, actions, targets

    def get_rate(self, episode, min_rate, max_rate, denominator):
        '''
        Calculates the logarithmically decreasing explore or learning rate.
        :param episode: the current episode
        :param min_rate: the minimum possible rate size
        :param max_rate: maximum rate size
        :param denominator: controls the rate of decay of the rate size
        :returns: the new rate size
        '''

        # input validation
        if not 0 <= min_rate <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= max_rate <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < denominator:
            raise ValueError("denominator needs to be above 0")

        rate = max(min_rate, min(max_rate, 1.0 - math.log10((episode + 1) / denominator)))
        return rate

    def train_q_net(self, q_net, inputs, targets, optimizer, criterion, epochs, batch_size=256, recurrent=True):
        '''
        Trains the Q network on the given inputs and targets.
        :param q_net: the Q network to train
        :param inputs: the inputs to the Q network
        :param targets: the targets for the Q network
        :param optimizer: the optimizer to use
        :param criterion: the loss function to use
        :param epochs: the number of epochs to train for
        :param batch_size: the batch size to use
        :param recurrent: whether to use the recurrent networks
        :return: the trained Q network
        '''
        
        if type(targets) in (list, tuple, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        batch_idxs = math.ceil(len(inputs[0]) / batch_size)
        
        for _ in range(epochs):
            # go through all the batches
            for batch_i in range(batch_idxs):
                start_idx = batch_i * batch_size
                end_idx = start_idx + batch_size

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                pred_values = self.forward_q_net(q_net=q_net, inputs=[inps[start_idx:end_idx] for inps in inputs], recurrent=recurrent)

                # calculate loss + credit assignment
                loss = criterion(pred_values, targets[start_idx:end_idx])
                loss.backward()

                # update model weights
                optimizer.step()

        return q_net
    
    def train_policy(self, inputs, epochs, recurrent=True):
        '''
        Train the policy network on the given inputs.
        :param inputs: the inputs to the policy network
        :param epochs: the number of epochs to train for
        :param recurrent: whether to use the recurrent networks
        :return: the trained policy network
        '''
        
        if recurrent:
            states, sequences = inputs
        else:
            states = inputs[0]
        
        batch_idxs = math.ceil(states.shape[0] / self.batch_size)
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []
            for batch_i in range(batch_idxs):
                start_idx = batch_i * self.batch_size
                end_idx = start_idx + self.batch_size

                self.policy_network_opt.zero_grad()

                ### run: policy network -> state-action pairs -> Q values
                pred_actions = self.forward_policy_net(
                    policy_net=self.policy_network,
                    inputs=[states[start_idx:end_idx], sequences[start_idx:end_idx]],
                    recurrent=recurrent,
                )
                # gradients flows from final Q-values through pred_actions to policy network's parameters
                q_net_inputs = [
                    torch.cat((
                        torch.tensor(states[start_idx:end_idx], dtype=torch.float32, device=self.device),
                        pred_actions
                    ), dim=1),
                    sequences[start_idx:end_idx]
                ]
                pred_values = self.forward_q_net(
                    q_net=self.Q1_network,
                    inputs=q_net_inputs,
                    recurrent=recurrent
                )

                ### calculate loss + credit assignment
                loss = self.policy_network_loss(pred_values)
                loss.backward()
                self.policy_network_opt.step()

                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

            # clear the gradients from both networks
            self.policy_network_opt.zero_grad()
            self.Q1_network_opt.zero_grad()

            # if fitted:
            #     if epoch == 0:
            #         best_weights = self.policy_network.get_weights()
            #         best = np.mean(batch_losses)
            #         wait = 0
            #     elif np.mean(batch_losses) < best:

            #         best_weights = self.policy_network.get_weights()
            #         wait = 0
            #         best = np.mean(batch_losses)
            #     else:
            #         wait += 1

            #     if wait >= patience:
            #         self.policy_network.set_weights(best_weights)
            #         break

        #print('Policy epochs: ', len(epoch_losses), epoch_losses[0], epoch_losses[-1])

        return epoch_losses

    def validate_on_train(model, dataloader, loss):
        model.eval()
        loss_total = 0

        with torch.no_grad():
            for data in dataloader:
                input = data[0]
                label = data[1]

                output = model(input.view(input.shape[0], -1))
                loss = loss(output, label)
                loss_total += loss.item()

        return loss_total / len(dataloader)

    def Q_update(self, recurrent=True, monte_carlo=False, policy=True, verbose=False, low_mem=True, epochs=1):
        '''
        Updates the Q network parameters.
        :param recurrent: whether to use the recurrent networks
        :param monte_carlo:
        :param policy: whether to update the policy network afterwards
        :param fitted:
        :param verbose:
        :param low_mem: whether to use low memory mode
        '''

        from time import time
        low_mem = False
        if low_mem:
            start = time()
            inputs, actions, targets = self.get_inputs_targets_low_mem(
                recurrent=recurrent, monte_carlo=monte_carlo)
            print('get_inputs_targets_low_mem: ', time() - start)
        else:
            start = time()
            inputs, actions, targets = self.get_inputs_targets(
                recurrent=recurrent, monte_carlo=monte_carlo)
            print('get_inputs_targets: ', time() - start)
        
        if recurrent:
            states, sequences = inputs
        else:
            states = inputs[0]
        
        q1_net_inputs = [np.concatenate((states, actions), axis=1), sequences] if recurrent else [np.concatenate((states, actions), axis=1)]
        self.Q1_network = self.train_q_net(
            q_net=self.Q1_network,
            inputs=q1_net_inputs,
            targets=targets,
            optimizer=self.Q1_network_opt,
            criterion=self.Q1_network_loss,
            epochs=epochs,
            batch_size=self.batch_size,
            recurrent=recurrent,
        )
        q2_net_inputs = [np.concatenate((states, actions), axis=1), sequences] if recurrent else [np.concatenate((states, actions), axis=1)]
        self.Q2_network = self.train_q_net(
            q_net=self.Q2_network,
            inputs=q2_net_inputs,
            targets=targets,
            optimizer=self.Q2_network_opt,
            criterion=self.Q2_network_loss,
            epochs=epochs,
            batch_size=self.batch_size,
            recurrent=recurrent,
        )

        if policy:
            epoch_losses = self.train_policy(
                inputs=[states, sequences] if recurrent else [states],
                epochs=epochs,
                recurrent=recurrent,
            )

        ### update target networks when we update the policy
        if policy and not monte_carlo:
            self.update_target_network(source=self.Q1_network, target=self.Q1_target, tau=self.polyak)
            self.update_target_network(source=self.Q2_network, target=self.Q2_target, tau=self.polyak)
            self.update_target_network(source=self.policy_network, target=self.policy_target, tau=self.polyak)

    def save_network(self, save_path):
        '''
        Saves networks to directory specified by save_path
        :param save_path: directory to save networks to
        '''
        
        torch.save(self.policy_network, os.path.join(save_path, "policy_network.pth"))
        torch.save(self.Q1_network, os.path.join(save_path, "Q1_network.pth"))
        torch.save(self.Q2_network, os.path.join(save_path, "Q2_network.pth"))

        torch.save(self.policy_target, os.path.join(save_path, "policy_target.pth"))
        torch.save(self.Q1_target, os.path.join(save_path, "Q1_target.pth"))
        torch.save(self.Q2_target, os.path.join(save_path, "Q2_target.pth"))

    def load_network(self, load_path, load_target_networks=False):
        '''
        Loads netoworks from directory specified by load_path.
        :param load_path: directory to load networks from
        :param load_target_networks: whether to load target networks
        '''
        
        self.policy_network = torch.load(os.path.join(load_path, "policy_network.pth"))
        self.policy_network_opt =  Adam(self.policy_network.parameters(), lr=self.pol_learning_rate)
        
        self.Q1_network = torch.load(os.path.join(load_path, "Q1_network.pth"))
        self.Q1_network_opt = Adam(self.Q1_network.parameters(), lr=self.val_learning_rate)
        
        self.Q2_network = torch.load(os.path.join(load_path, "Q2_network.pth"))
        self.Q2_etwork_opt = Adam(self.Q2_network.parameters(), lr=self.val_learning_rate)
        
        if load_target_networks:
            self.policy_target = torch.load(os.path.join(load_path, "policy_target.pth"))
            self.Q1_target = torch.load(os.path.join(load_path, "Q1_target.pth"))
            self.Q2_target = torch.load(os.path.join(load_path, "Q2_target.pth"))
        else:
            print("[WARNING] Not loading target networks")

    def reset_weights(self, policy=True):
        '''
        Reinitialises weights to random values.
        :param policy: whether to reinitialise policy network
        '''
        del self.Q1_network
        del self.Q2_network
        if policy:
            del self.policy_network
        gc.collect()

        self.Q1_network = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in self.val_module_specs
        ])
        self.Q1_network_opt = Adam(self.Q1_network.parameters(), lr=self.val_learning_rate)
        self.Q1_network_loss = nn.MSELoss()

        self.Q2_network = nn.ModuleList([
            self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
            for mod_specs in self.val_module_specs
        ])
        self.Q2_network_opt = Adam(self.Q2_network.parameters(), lr=self.val_learning_rate)
        self.Q2_network_loss = nn.MSELoss()

        if policy:
            ### initialise policy networks
            self.policy_network = nn.ModuleList([
                self.initialise_network(input_size=mod_specs["input_size"], layer_specs=mod_specs["layers"], init_optimizer=False)
                for mod_specs in self.pol_module_specs
            ])
            self.policy_network_opt =  Adam(self.policy_network.parameters(), lr=self.pol_learning_rate)
            self.policy_network_loss = lambda predicted_action_values: -torch.mean(predicted_action_values)

    def update_target_network(self, source, target, tau):
        '''
        Updates the target network from the source network using Polyak averaging.
        :param source: source network
        :param target: target network
        :param tau: Polyak averaging parameter
        :return: updated target network
        '''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data * tau + target_param.data * (1.0 - tau))
        return target
