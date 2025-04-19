import numpy as np
import os
import tensorflow as tf
from rlsp.agents.rlsp_agent import RLSPAgent

class RLSPTD3Agent(RLSPAgent):
    def __init__(self):
        super().__init__()
        self.replay_buffer = []  # Properly initialize replay buffer in the constructor
        self.actor = None
        self.critic_1 = None
        self.critic_2 = None
        self.target_actor = None
        self.target_critic_1 = None
        self.target_critic_2 = None
        self.config = None
        self.step_count = 0


    def create(self, env, config, result, logger):
        self.env = env
        self.config = config
        self.result = result
        self.logger = logger
        self.buffer_size = config.get('mem_limit', 20000)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('target_model_update', 0.0001)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.noise_std = config.get('rand_sigma', 0.2)
        self.update_freq = config.get('update_freq', 2)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.actor = self._build_actor(obs_dim, act_dim)
        self.target_actor = self._build_actor(obs_dim, act_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.critic_1 = self._build_critic(obs_dim, act_dim)
        self.critic_2 = self._build_critic(obs_dim, act_dim)
        self.target_critic_1 = self._build_critic(obs_dim, act_dim)
        self.target_critic_2 = self._build_critic(obs_dim, act_dim)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        return self

    def _build_actor(self, obs_dim, act_dim):
        inputs = tf.keras.Input(shape=(obs_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(act_dim, activation='tanh')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def _build_critic(self, obs_dim, act_dim):
        obs_input = tf.keras.Input(shape=(obs_dim,))
        act_input = tf.keras.Input(shape=(act_dim,))
        x = tf.keras.layers.Concatenate()([obs_input, act_input])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model([obs_input, act_input], outputs)
        return model

    def act(self, observation):
        obs = np.expand_dims(observation, axis=0).astype(np.float32)  # Cast to float32
        action = self.actor.predict(obs)[0]
        if self.config.get('training', True):
            noise = np.random.normal(0, self.noise_std, size=action.shape).astype(np.float32)  # Cast noise
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
        return action

    def test(self, env, episodes=1, verbose=0, episode_steps=200, callbacks=None):
        self.logger.info("Starting test phase")
        # Ensure config is a dictionary; use default if None
        if self.config is None:
            self.logger.warning("self.config is None; using default config with training=False")
            self.config = {'training': False}
        else:
            self.config['training'] = False
        for episode in range(episodes):
            observation = env.reset()
            episode_reward = 0
            for step in range(episode_steps):
                action = self.act(observation)
                next_observation, reward, done, info = env.step(action)
                episode_reward += reward
                observation = next_observation
                if done:
                    break
            
            # Log episode results if verbose
            if verbose > 0:
                self.logger.info(f"Test Episode {episode + 1}/{episodes} - Reward: {episode_reward}")
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_episode_end(episode, {'reward': episode_reward})
        
        self.logger.info("Finished test phase")

    def save_weights(self, directory, overwrite=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.actor.save_weights(os.path.join(directory, 'actor_weights.h5'), overwrite=overwrite)
        self.critic_1.save_weights(os.path.join(directory, 'critic_1_weights.h5'), overwrite=overwrite)
        self.critic_2.save_weights(os.path.join(directory, 'critic_2_weights.h5'), overwrite=overwrite)
        self.target_actor.save_weights(os.path.join(directory, 'target_actor_weights.h5'), overwrite=overwrite)
        self.target_critic_1.save_weights(os.path.join(directory, 'target_critic_1_weights.h5'), overwrite=overwrite)
        self.target_critic_2.save_weights(os.path.join(directory, 'target_critic_2_weights.h5'), overwrite=overwrite)


    def load_weights(self, weights_file):
        actor_weights = f"{weights_file}/actor_weights.h5"
        if not os.path.exists(actor_weights):
            raise FileNotFoundError(f"Weights file not found: {actor_weights}")
        self.actor.load_weights(actor_weights)
        self.critic_1.load_weights(f"{weights_file}/critic_1_weights.h5")
        self.critic_2.load_weights(f"{weights_file}/critic_2_weights.h5")
        self.target_actor.load_weights(f"{weights_file}/target_actor_weights.h5")
        self.target_critic_1.load_weights(f"{weights_file}/target_critic_1_weights.h5")
        self.target_critic_2.load_weights(f"{weights_file}/target_critic_2_weights.h5")
        self.logger.info(f"Successfully loaded weights from {weights_file}")

    def train(self, observation, action, reward, next_observation, done):
        self.replay_buffer.append((observation, action, reward, next_observation, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        if len(self.replay_buffer) >= self.batch_size:
            batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            obs, acts, rews, next_obs, dones = zip(*[self.replay_buffer[i] for i in batch])
            obs = np.array(obs, dtype=np.float32)  # Cast to float32
            acts = np.array(acts, dtype=np.float32)  # Cast to float32
            rews = np.array(rews, dtype=np.float32)  # Cast to float32
            next_obs = np.array(next_obs, dtype=np.float32)  # Cast to float32
            dones = np.array(dones, dtype=np.float32)  # Cast to float32

            target_actions = self.target_actor.predict(next_obs)
            noise = np.random.normal(0, self.noise_std, size=target_actions.shape).astype(np.float32)  # Cast noise
            target_actions = np.clip(target_actions + noise, self.env.action_space.low, self.env.action_space.high)

            target_q1 = self.target_critic_1.predict([next_obs, target_actions])
            target_q2 = self.target_critic_2.predict([next_obs, target_actions])
            target_q = np.minimum(target_q1, target_q2)
            targets = rews + self.gamma * (1 - dones) * target_q

            with tf.GradientTape() as tape:
                q1 = self.critic_1([obs, acts])
                loss1 = tf.reduce_mean(tf.square(targets - q1))
            grads1 = tape.gradient(loss1, self.critic_1.trainable_variables)
            self.critic_1_optimizer.apply_gradients(zip(grads1, self.critic_1.trainable_variables))

            with tf.GradientTape() as tape:
                q2 = self.critic_2([obs, acts])
                loss2 = tf.reduce_mean(tf.square(targets - q2))
            grads2 = tape.gradient(loss2, self.critic_2.trainable_variables)
            self.critic_2_optimizer.apply_gradients(zip(grads2, self.critic_2.trainable_variables))

            self.step_count += 1
            if self.step_count % self.update_freq == 0:
                with tf.GradientTape() as tape:
                    new_actions = self.actor(obs)
                    q_values = self.critic_1([obs, new_actions])
                    actor_loss = -tf.reduce_mean(q_values)
                grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                for source, target in [
                    (self.actor, self.target_actor),
                    (self.critic_1, self.target_critic_1),
                    (self.critic_2, self.target_critic_2)
                ]:
                    source_weights = source.get_weights()
                    target_weights = target.get_weights()
                    for i in range(len(source_weights)):
                        target_weights[i] = self.tau * source_weights[i] + (1 - self.tau) * target_weights[i]
                    target.set_weights(target_weights)
