import copy
import gc
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


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



class RT3D_agent():
    '''
    class that implements the RT3D agent
    '''
    def __init__(self, val_layer_sizes, pol_layer_sizes, gamma = 1, val_learning_rate = 0.001, pol_learning_rate = 0.001, policy_act = tf.keras.activations.linear):
        '''
        initialises the agent
        :param val_layer_sizes: layer sizes fo rthe Q networks
        :param pol_layer_sizes: layer sizes for the policy networks
        :param gamma: discount rate
        :param val_learning_rate:
        :param pol_learning_rate:
        :param policy_act: activation for the output of the policy network
        '''
        self.layer_sizes = pol_layer_sizes
        self.val_layer_sizes = val_layer_sizes
        self.val_learning_rate = val_learning_rate
        self.memory = []
        self.gamma = gamma
        self.scale = 1
        self.polyak = 0.995
        self.batch_size = 256
        self.policy_network = self.initialise_network(pol_layer_sizes, out_act = policy_act, scale = self.scale)
        self.policy_opt = keras.optimizers.Adam(learning_rate=pol_learning_rate)
        self.policy_act = policy_act

        self.policy_target = self.initialise_network(pol_layer_sizes, out_act=policy_act, scale=self.scale)
        self.policy_target_opt = keras.optimizers.Adam(learning_rate=pol_learning_rate)

        self.Q1_network = self.initialise_network(val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=val_learning_rate)
        self.Q1_network.compile(optimizer=opt, loss='mean_squared_error')

        self.Q1_target = self.initialise_network(val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=val_learning_rate)
        self.Q1_target.compile(optimizer=opt, loss='mean_squared_error')

        self.Q2_network = self.initialise_network(val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=val_learning_rate)
        self.Q2_network.compile(optimizer=opt, loss='mean_squared_error')

        self.Q2_target = self.initialise_network(val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=val_learning_rate)
        self.Q2_target.compile(optimizer=opt, loss='mean_squared_error')

        self.values = []
        self.actions = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequences = []
        self.next_sequences = []
        self.all_returns = []

    def initialise_network(self, layer_sizes, out_act = tf.keras.activations.linear, scale = 1.):
        '''
        initialises a neural network
        :param layer_sizes: the layer sizes of each part of the network
        :param out_act: activation of the output layer
        :param scale: scale of the output (so network output can be between 0 and 1)
        :return:
        '''
        input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes

        S_input = keras.Input(shape = (input_size,), name = "S_input")

        if sequence_size != 0:
            sequence_input = keras.Input(shape = (None,sequence_size), name = 'sequence_input')

            rec_out = sequence_input
            for i, rec_size in enumerate(rec_sizes):

                if i == len(rec_sizes) -1:
                    rec_out = layers.GRU(rec_size)(rec_out)
                else:
                    rec_out = layers.GRU(rec_size, input_shape = (None,sequence_size), return_sequences=True)(rec_out)

            concat = layers.concatenate([S_input, rec_out])
        else:
            concat = S_input

        hl = concat

        for i, hl_size in enumerate(hidden_sizes):
            hl = layers.Dense(hl_size,activation=tf.nn.relu, name = 'hidden_' + str(i))(hl)

        out =  layers.Dense(layer_sizes[-1], name='mu', activation=out_act)(hl)
        out = layers.Lambda(lambda x: x*scale)(out)

        if sequence_size != 0:
            network = keras.Model(
                inputs=[S_input, sequence_input],
                outputs=[out]
            )
        else:
            network = keras.Model(
                inputs=[S_input],
                outputs=[out]
            )
        return network


    def get_actions_dist(self, inputs, explore_rate, test_episode = False, recurrent = True):
        '''
        gets actions by adding random noise to the actions
        :param inputs: RL inputs
        :param explore_rate:
        :param test_episode:
        :param recurrent:
        :return:
        '''

        if recurrent:
            states, sequences = inputs
            sequences = pad_sequences(sequences, maxlen=self.max_length, dtype='float64')


            actions = self.policy_network([np.array(states), sequences])
        else:
            states = inputs[0]
            actions = self.policy_network([np.array(states)])
        actions = np.array(actions)

        if test_episode:

            actions[:-1] += np.random.normal(0, explore_rate,size = actions[:-1].shape)

        else:
            actions += np.random.normal(0, explore_rate, size=actions.shape)
        actions = np.clip(actions, self.action_bounds[0], self.action_bounds[1])

        return actions

    def get_actions(self, inputs, explore_rate, test_episode = False, recurrent = True):
        '''
        gets actions by adding choosing random action between the min and max bounds with probablilty explore_rate
        :param inputs: RL inputs
        :param explore_rate:
        :param test_episode:
        :param recurrent:
        :return:
        '''

        states, sequences = inputs

        if test_episode:
            rng = np.random.random(len(states)-1)
        else:
            rng = np.random.random(len(states))

        explore_inds = np.where(rng < explore_rate)[0]
        exploit_inds = np.where(rng >= explore_rate)[0]

        if test_episode: exploit_inds = np.append(exploit_inds, len(states)-1)

        explore_actions = np.random.uniform(self.action_bounds[0], self.action_bounds[1], size = (len(explore_inds), self.layer_sizes[-1]))


        actions = np.zeros((len(states), self.layer_sizes[-1]), dtype='float64')

        if len(exploit_inds) > 0:

            sequences = pad_sequences(sequences, maxlen=self.max_length, dtype='float64')
            exploit_actions = self.policy_network([np.array(states)[exploit_inds], np.array(sequences)[exploit_inds]])

            exploit_actions += np.random.normal(0, explore_rate*self.std*2, size=exploit_actions.shape)

            exploit_actions = np.clip(exploit_actions, self.action_bounds[0], self.action_bounds[1])

            actions[exploit_inds] = exploit_actions


        actions[explore_inds] = explore_actions

        exploit_flags = np.zeros((len(states)), dtype=np.int32) #just for interest
        exploit_flags[exploit_inds] = 1

        return actions#, exploit_flags

    def get_action(self, s, explore_rate):
        a = self.policy_network(s)
        noise = np.random.normal(0, explore_rate,size = a.shape)

        a += noise
        a = np.clip(a, -1, 1)

        return a


    def get_inputs_targets(self, recurrent = True, monte_carlo = False, fitted = False):
        '''
        assembles the Q learning inputs and trgets from agents memory
        :param recurrent:
        :param monte_carlo:
        :param fitted:
        :return:
        '''

        # iterate over all exprienc in memory and create fitted Q targets
        for i, trajectory in enumerate(self.memory):

            e_rewards = []
            sequence = [[0]*self.layer_sizes[1]]
            for j, transition in enumerate(trajectory):
                self.sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done = transition

                sequence.append(np.concatenate((state, action)))
                #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10
                self.next_sequences.append(copy.deepcopy(sequence))
                self.states.append(state)
                self.next_states.append(next_state)
                self.actions.append(action)
                self.rewards.append(reward)
                e_rewards.append(reward)
                self.dones.append(done)

            if monte_carlo:
                e_values = [e_rewards[-1]]

                for i in range(2, len(e_rewards) + 1):
                    e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
                self.all_returns.extend(e_values)


        if recurrent:
            padded = pad_sequences(self.sequences, maxlen = self.max_length, dtype='float64')[:self.mem_size]
            next_padded = pad_sequences(self.next_sequences, maxlen=self.max_length, dtype='float64')[:self.mem_size]


        next_states = np.array(self.next_states, dtype=np.float64)[:self.mem_size]
        rewards = np.array(self.rewards).reshape(-1, 1)[:self.mem_size]
        dones = np.array(self.dones).reshape(-1, 1)[:self.mem_size]
        states = np.array(self.states)[:self.mem_size]
        actions = np.array(self.actions)[:self.mem_size]
        all_returns = np.array(self.all_returns)[:self.mem_size]

        self.memory = []  # reset memory after this information has been extracted


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
            pass

        elif not fitted:
            # take random sample

            sample_size = int(self.batch_size*10)

            indices = np.random.randint(max(0, states.shape[0] - self.mem_size), states.shape[0], size=(sample_size))

            states = states[indices]

            next_states = next_states[indices]
            actions = actions[indices]
            rewards = rewards[indices]
            dones = dones[indices]

            if recurrent:
                padded = padded[indices]
                next_padded = next_padded[indices]

        #values = self.predict([states, padded])

        if monte_carlo:
            targets = all_returns
        else:

            if fitted:
                Q1_target = self.Q1_network
                Q2_target = self.Q2_network
                policy_target = self.policy_network
            else:
                Q1_target = self.Q1_target
                Q2_target = self.Q2_target
                policy_target = self.policy_target

            next_actions = policy_target([next_states, next_padded]) if recurrent else self.policy_target([next_states])

            # target policy smoothing

            noise = np.clip(np.random.normal( 0, self.std, next_actions.shape), self.noise_bounds[0], self.noise_bounds[1])


            next_actions = np.clip(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

            #next_actions = np.vstack((actions[1:], actions[0])) #sarsa
            Q1 = Q1_target.predict([tf.concat((next_states, next_actions), 1), next_padded]) if recurrent else Q1_target.predict([tf.concat((next_states, next_actions), 1)])
            Q2 = Q2_target.predict([tf.concat((next_states, next_actions), 1), next_padded]) if recurrent else Q2_target.predict([tf.concat((next_states, next_actions), 1)])

            next_values = np.minimum(Q1, Q2)
            #next_values = Q1
            targets = rewards + self.gamma*(1-dones)*next_values




        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]
        actions = actions[randomize]

        if recurrent: padded = padded[randomize]

        targets = targets[randomize]

        inputs = [states, padded] if recurrent else [states]
        #print('inputs, actions, targets', inputs[0].shape, actions.shape, targets.shape)

        return inputs, actions, targets

    def get_inputs_targets_low_mem(self, recurrent = True, monte_carlo = False, fitted = False):
        '''
        assembles the Q learning inputs and trgets from agents memory, uses less memory but is slower
        :param recurrent:
        :param monte_carlo:
        :param fitted:
        :return:
        '''

        #TODO:: enable all the options here
        self.memory = self.memory[-self.mem_size:]

        sample_size = int(self.batch_size*10)

        indices = np.random.randint(0, min(self.mem_size, len(self.memory)), size=(sample_size))


        # sample = np.array(self.memory)[indices]
        sample = []
        for i in indices:
            sample.append(self.memory[i])
        #print(sample)

        sequences = []
        next_sequences = []
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        # iterate over all exprienc in memory and create fitted Q targets
        for i, trajectory in enumerate(sample):

            e_rewards = []
            sequence = [[0] * self.layer_sizes[1]]
            for j, transition in enumerate(trajectory):
                sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done = transition

                sequence.append(np.concatenate((state, action)))
                next_sequences.append(copy.deepcopy(sequence))
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                e_rewards.append(reward)
                dones.append(done)



        padded = pad_sequences(sequences, maxlen=self.max_length, dtype='float64')
        next_padded = pad_sequences(next_sequences, maxlen=self.max_length, dtype='float64')

        next_states = np.array(next_states, dtype=np.float64)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        states = np.array(states)
        actions = np.array(actions)




        #values = self.predict([states, padded])

        next_actions = self.policy_target([next_states, next_padded]) if recurrent else self.policy_target([next_states])

        # target policy smoothing

        noise = np.clip(np.random.normal( 0, self.std, next_actions.shape), self.noise_bounds[0], self.noise_bounds[1])


        next_actions = np.clip(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

        #next_actions = np.vstack((actions[1:], actions[0])) #sarsa
        Q1 = self.Q1_target.predict([tf.concat((next_states, next_actions), 1), next_padded], verbose=0) if recurrent else self.Q1_target.predict([tf.concat((next_states, next_actions), 1)], verbose=0)
        Q2 = self.Q2_target.predict([tf.concat((next_states, next_actions), 1), next_padded], verbose=0) if recurrent else sel.fQ2_target.predict([tf.concat((next_states, next_actions), 1)], verbose=0)

        next_values = np.minimum(Q1, Q2)

        targets = rewards + self.gamma*(1-dones)*next_values

        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]
        actions = actions[randomize]

        padded = padded[randomize]

        targets = targets[randomize]

        inputs = [states, padded]
        #print('inputs, actions, targets', inputs[0].shape, actions.shape, targets.shape)

        return inputs, actions, targets

    def get_rate(self, episode, MIN_RATE, MAX_RATE, denominator):
        '''
        Calculates the logarithmically decreasing explore or learning rate
        :param episode: the current episode
        :param MIN_LEARNING_RATE: the minimum possible step size
        :param MAX_LEARNING_RATE: maximum step size
        :param denominator: controls the rate of decay of the step size
        :returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= MIN_RATE <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= MAX_RATE <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < denominator:
            raise ValueError("denominator needs to be above 0")

        rate = max(MIN_RATE, min(MAX_RATE, 1.0 - math.log10((episode + 1) / denominator)))

        return rate

    def Q_update(self, recurrent = True, monte_carlo =False, policy = True, fitted = True, verbose = False, low_mem = True):
        '''
        Updates the Q learning parameters
        :param recurrent:
        :param monte_carlo:
        :param policy:
        :param fitted:
        :param verbose:
        :param low_mem:
        :return:
        '''
        if low_mem:
            inputs, actions, targets = self.get_inputs_targets_low_mem(recurrent=recurrent, monte_carlo=monte_carlo,
                                                                       fitted=fitted)
        else:
            inputs, actions, targets = self.get_inputs_targets(recurrent=recurrent, monte_carlo=monte_carlo,
                                                               fitted=fitted)

        if recurrent:
            states, sequences = inputs
        else:
            states = inputs[0]

        if fitted:

            epochs = 500
            patience = 10


            self.reset_weights(policy=policy)
            callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=patience, restore_best_weights=True)
            callbacks = [callback]
        else:

            epochs = 1

            patience = 1
            callbacks = []

        if recurrent:
            history1 = self.Q1_network.fit([tf.concat((states, actions), 1), sequences], targets, epochs = epochs, verbose = verbose, validation_split =0., batch_size=self.batch_size, callbacks = callbacks)
            history2 = self.Q2_network.fit([tf.concat((states, actions), 1), sequences], targets, epochs = epochs, verbose = verbose, validation_split =0., batch_size=self.batch_size, callbacks = callbacks)
        else:
            history1 = self.Q1_network.fit([tf.concat((states, actions), 1)], targets, epochs = epochs, verbose = False, validation_split =0., batch_size=self.batch_size, callbacks = callbacks)
            history2 = self.Q2_network.fit([tf.concat((states, actions), 1)], targets, epochs = epochs, verbose = False, validation_split =0., batch_size=self.batch_size, callbacks = callbacks)


        if policy:

            batches = math.ceil(states.shape[0]/self.batch_size)

            epoch_losses = []
            for epoch in range(epochs):
                batch_losses = []
                for batch in range(batches):

                    start = batch*self.batch_size
                    end = start + self.batch_size

                    with tf.GradientTape() as tape:
                        pred_actions = self.policy_network([states[start:end], sequences[start:end]]) if recurrent else self.policy_network([states[start:end]])
                        pred_values = self.Q1_network([tf.concat((states[start:end], pred_actions), 1), sequences[start:end]]) if recurrent else self.Q1_network([tf.concat((states[start:end], pred_actions), 1)])

                        loss = -tf.math.reduce_mean(pred_values)

                    policy_grad = tape.gradient(loss, self.policy_network.trainable_variables)

                    self.policy_opt.apply_gradients(zip(policy_grad, self.policy_network.trainable_variables))

                    batch_losses.append(loss)

                epoch_losses.append(np.mean(batch_losses))

                if fitted:
                    if epoch == 0:
                        best_weights = self.policy_network.get_weights()
                        best = np.mean(batch_losses)
                        wait = 0
                    elif np.mean(batch_losses) < best:

                        best_weights = self.policy_network.get_weights()
                        wait = 0
                        best = np.mean(batch_losses)
                    else:
                        wait += 1

                    if wait >= patience:

                        self.policy_network.set_weights(best_weights)

                        break

            #print('Policy epochs: ', len(epoch_losses), epoch_losses[0], epoch_losses[-1])

        if not fitted and not monte_carlo and policy: # update target nbetworks when we update the policy

            self.update_target_network(self.Q1_network, self.Q1_target, self.polyak)
            self.update_target_network(self.Q2_network, self.Q2_target, self.polyak)
            self.update_target_network(self.policy_network, self.policy_target, self.polyak)

        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    def save_network(self, save_path): # tested
        '''
        saves networks to file
        :param save_path:
        :return:
        '''
        #print(self.network.layers[1].get_weights())
        self.policy_network.save(os.path.join(save_path,'policy_network.h5'))
        self.Q1_network.save(os.path.join(save_path, 'Q1_network.h5'))
        self.Q2_network.save(os.path.join(save_path, 'Q2_network.h5'))

        self.policy_target.save(os.path.join(save_path, 'policy_target.h5'))
        self.Q1_target.save(os.path.join(save_path, 'Q1_target.h5'))
        self.Q2_target.save(os.path.join(save_path,'Q2_target.h5'))

    def load_network(self, load_path): #tested
        '''
        load netowkrs from files
        :param load_path:
        :return:
        '''
        print('LOADING NETWORKS, UNCOMMENT HERE TO ALSO LOAD TARGET NETWORKS')


        try:
            self.policy_network = keras.models.load_model(os.path.join(load_path, 'policy_network.h5')) # sometimes this crashes, apparently a bug in keras
            self.Q1_network = keras.models.load_model(os.path.join(load_path, 'Q1_network.h5'))
            self.Q2_network = keras.models.load_model(os.path.join(load_path, 'Q2_network.h5'))
            self.policy_target = keras.models.load_model(os.path.join(load_path,'policy_target.h5'))  # sometimes this crashes, apparently a bug in keras
            self.Q1_target = keras.models.load_model(os.path.join(load_path,'Q1_target.h5'))
            self.Q2_target = keras.models.load_model(os.path.join(load_path, 'Q2_target.h5'))
        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.policy_network.load_weights(os.path.join(load_path, 'policy_network.h5')) # this requires model to be initialised exactly the same
            self.Q1_network.load_weights(os.path.join(load_path, 'Q1_network.h5'))
            self.Q2_network.load_weights(os.path.join(load_path, 'Q2_network.h5'))
            self.policy_target.load_weights(os.path.join(load_path, 'policy_target.h5'))  # this requires model to be initialised exactly the same
            self.Q1_target.load_weights(os.path.join(load_path,'Q1_target.h5'))
            self.Q2_target.load_weights(os.path.join(load_path, 'Q2_target.h5'))

    def reset_weights(self, policy = True):
        '''
        Reinitialises weights to random values
        '''
        #sess = tf.keras.backend.get_session()
        #sess.run(tf.global_variables_initializer())
        del self.Q1_network
        del self.Q2_network
        if policy: del self.policy_network
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.Q1_network = self.initialise_network(self.val_layer_sizes)
        self.Q2_network = self.initialise_network(self.val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=self.val_learning_rate)  # no nfitted methods
        self.Q1_network.compile(optimizer=opt, loss='mean_squared_error')
        self.Q2_network.compile(optimizer=opt, loss='mean_squared_error')
        if policy: self.policy_network = self.initialise_network(self.layer_sizes, out_act=self.policy_act, scale=self.scale)

    def update_target_network(self, source, target,tau):
        '''
        updates the target networks using Polyack averaging
        :param source:
        :param target:
        :param tau:
        :return:
        '''
        source_weights  = source.variables
        target_weights = target.variables

        for (source, target) in zip(source_weights, target_weights):
            target.assign(source * tau + target * (1 - tau))