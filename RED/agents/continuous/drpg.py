import copy

import numpy as np
import tensorflow as tf
from keras.api._v2 import keras


class DRPG_agent():
    def __init__(self, layer_sizes, learning_rate = 0.001, critic=True):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.

        self.critic = critic
        if critic:
            self.critic_network = self.initialise_network(layer_sizes, critic_nw=True)
            self.critic_network.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss='mean_squared_error')
        self.actor_network = self.initialise_network(layer_sizes)
        self.opt = keras.optimizers.Adam(learning_rate=learning_rate)

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


    def initialise_network(self, layer_sizes, critic_nw = False):

        '''
        Creates Q network for value function approximation
        '''
        input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes


        S_input = keras.Input(shape = (input_size,), name = "S_input")
        sequence_input = keras.Input(shape = (None,sequence_size), name = 'sequence_input')


        rec_out = sequence_input
        for i, rec_size in enumerate(rec_sizes):

            if i == len(rec_sizes) -1:
                rec_out = layers.GRU(rec_size)(rec_out)
            else:
                rec_out = layers.GRU(rec_size, input_shape = (None,sequence_size), return_sequences=True)(rec_out)



        concat = layers.concatenate([S_input, rec_out])

        hl = concat

        for i, hl_size in enumerate(hidden_sizes):
            hl = layers.Dense(hl_size,activation=tf.nn.relu, name = 'hidden_' + str(i))(hl)


        if critic_nw:

            values = layers.Dense(1, name='mu')(hl)


            network = keras.Model(
                inputs=[S_input, sequence_input],
                outputs=[values]
            )
        else:
            mu = layers.Dense(output_size, name = 'mu', activation=tf.nn.sigmoid)(hl)
            log_std = layers.Dense(output_size, name = 'log_std')(hl)

            network = keras.Model(
                inputs = [S_input, sequence_input],
                outputs = [mu, log_std]
            )

        return network


    def get_actions(self, inputs):

        states, sequences = inputs


        sequences = pad_sequences(sequences, maxlen=self.max_length, dtype='float64')


        mu, log_std = self.actor_network.predict([np.array(states), sequences])

        print('mu log_std',mu[0], log_std[0])


        actions = mu + tf.multiply(tf.random.normal(tf.shape(mu)), tf.exp(log_std))
        #print('actions',actions[0])

        return actions

    def loss(self, inputs, actions, returns):
        # Obtain mu and sigma from actor network
        mu, log_std = self.actor_network(inputs)

        # Compute log probability
        log_probability = self.log_probability(actions, mu, log_std)
        print('log probability', log_probability.shape)
        print('returns:', returns.shape)
        # Compute weighted loss
        loss_actor = - tf.reduce_mean(tf.multiply(returns, log_probability))
        print('loss actor', loss_actor.shape)
        return loss_actor


    def log_probability(self, actions, mu, log_std):


        EPS = 1e-8
        pre_sum = -0.5 * (((actions - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))

        print('pre sum', pre_sum.shape)
        return tf.reduce_sum(pre_sum, axis=1)


    def policy_update(self):

        inputs, actions, returns = self.get_inputs_targets()

        print(returns.shape)
        if self.critic:

            expected_returns = self.critic_network.predict(inputs)

            returns -= expected_returns.reshape(-1)
            print(expected_returns.reshape(-1).shape)
            self.critic_network.fit(inputs, returns, epochs = 1)

        with tf.GradientTape() as tape:
            loss = self.loss(inputs, actions, returns)
            grads = tape.gradient(loss, self.actor_network.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.actor_network.trainable_variables))


    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        '''
                gets fitted Q inputs and calculates targets for training the Q-network for episodic training
                '''

        # iterate over all exprienc in memory and create fitted Q targets
        for i, trajectory in enumerate(self.memory):

            e_rewards = []
            sequence = [[0]*self.layer_sizes[1]]
            for j, transition in enumerate(trajectory):
                self.sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done, u = transition
                sequence.append(np.concatenate((state, u/1)))
                #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10
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

        padded = pad_sequences(self.sequences, maxlen = 11, dtype='float64')
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


    def save_network(self, save_path): # tested
        #print(self.network.layers[1].get_weights())
        self.actor_network.save(save_path + '/saved_network.h5')

    def load_network(self, load_path): #tested
        try:
            self.actor_network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras

        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.actor_network.load_weights(load_path+ '/saved_network.h5') # this requires model to be initialised exactly the same
