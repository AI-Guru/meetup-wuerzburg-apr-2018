"""
Namaste!

This is a very rudimentary implementation of Q-Learning for ViZDoom. It is
way beyond perfect. But it is a good starting point! Feel free to extendself.

Best,
Tristan Behrens
"""

from vizdoom import *
import random
import time
import keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

# Create an instance of the Doom game.
game = DoomGame()
game.load_config("scenarios/basic.cfg")
game.set_screen_format(ScreenFormat.GRAY8)
game.init()

# Hyper parameters.
epochs = 20 # Number of epochs to train.
learning_steps_per_epoch = 2000 # Number of learning steps per epoch.
screen_shape = (30, 45, 1) # The target shape of the screen.
#model_type = "dense" # A fully connected network.
model_type = "conv2d" # A convolutional neural network.

# These ere the actions.
action_none = [0, 0, 0]
action_shoot = [0, 0, 1]
action_left = [1, 0, 0]
action_right = [0, 1, 0]
actions = [action_shoot, action_left, action_right]
action_length = len(actions)


class DeepLearningAgent:
    """ This is the Deep Learning agent. """

    def __init__(self, model_type, screen_shape, action_length, actions):
        """ Initializes an instance of the agent. """

        # Set all parameters.
        self.screen_shape = screen_shape # The target screen shape.
        self.action_length = action_length # The length of the actions.
        self.actions = actions # The actions themselves.

        # Set the independent parameters.
        self.memory = deque(maxlen=2000) # Size of memory.
        self.gamma = 0.95    # Discount rate for learning.
        self.epsilon = 1.0  # Exploration rate, 1.0 is random, 0.0 is prediction.
        self.epsilon_min = 0.01 # Minimum the exploration rate cools down to.
        self.epsilon_decay = 0.9995 # Decay of the exploraion rate per step.
        self.learning_rate = 0.001 # Learning rate of the optimizer.

        # Create the model.
        self.model = self.build_model(model_type)


    def build_model(self, model_type):
        """ Builds the model according to a given type. """

        # Builds a simple fully connected network.
        if model_type == "dense":
            model = models.Sequential()
            model.add(layers.Flatten(input_shape=self.screen_shape))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(self.action_length, activation='linear'))

        # Builds a convolutional neural network.
        elif model_type == "conv2d":
            model = models.Sequential()
            model.add(layers.Conv2D(8, (6, 6), strides=(3, 3), input_shape=self.screen_shape))
            model.add(layers.Conv2D(8, (3, 3), strides=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(125, activation='relu'))
            model.add(layers.Dense(self.action_length, activation='linear'))

        # Compile and render the model.
        model.compile(
            loss='mse',
            optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()

        return model


    def remember(self, screen, action, reward, next_screen, done):
        """ Stores a state transition into memory. """
        assert screen.shape == self.screen_shape
        assert next_screen.shape == self.screen_shape
        self.memory.append((screen, action, reward, next_screen, done))


    def act(self, screen):
        """ Yields an action. Either a random or a predicted one. """

        # Return a random action.
        if np.random.rand() <= self.epsilon: # TODO use an exploration rate here
            return random.choice(self.actions)

        # Predict an action.
        else:
            screen = np.expand_dims(screen, axis=0)
            act_values = self.model.predict(screen)
            max_index = np.argmax(act_values[0])
            return self.actions[max_index]


    def replay(self, batch_size):
        """ Replays from memory and trains network. """

        # Train a mini batch.
        mini_batch = random.sample(self.memory, batch_size)

        for screen, action, reward, next_screen, done in mini_batch:
            screen = np.expand_dims(screen, axis=0)
            next_screen = np.expand_dims(next_screen, axis=0)

            target = reward
            if not done:
                prediction = self.model.predict(next_screen)[0]
                target = (reward + self.gamma * np.amax(prediction))
            target_f = self.model.predict(screen)
            target_f[0][action] = target
            self.model.fit(screen, target_f, epochs=1, verbose=0)

        # Let the exploration-rate decay.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, path):
        """ Loads the model from a path. """
        self.model.load_weights(path)


    def save(self, path):
        """ Saves the model to a path. """
        self.model.save_weights(path)


def main():
    """ This is the main method. """

    # Create the agent with its parameters.
    agent = DeepLearningAgent(model_type, screen_shape, action_length, actions)


    # Do the training.
    batch_size = 32 #  Size of the mini-batches for training
    done = True # Is the game episode done?
    for epoch in range(epochs):
        game.new_episode()

        for _ in range(learning_steps_per_epoch):

            # The game episode is done. Proceed properly.
            if done == True:
                done = False
                game.new_episode()
                state = game.get_state()
                screen = state.screen_buffer
                screen = transform_screen_buffer(screen, screen_shape)
                continue

            # Perform one step. Get an action, execute it, get the reward. Get
            # the next state.
            action = agent.act(screen)
            reward = game.make_action(action)
            next_state = game.get_state()

            # Make sure that the reset works.
            if game.is_episode_finished():
                done = True
                next_screen = screen
            else:
                next_screen = next_state.screen_buffer
                next_screen = transform_screen_buffer(next_screen, screen_shape)

            # Let the agent remember the trainsition.
            agent.remember(screen, action, reward, next_screen, done)
            screen = next_screen

            # Do the training.
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


def transform_screen_buffer(screen_buffer, target_shape):
    """ Transforms the screen buffer for the neural network. """

    # If it is RGB, swap the axes.
    if screen_buffer.ndim == 3:
        screen_buffer = np.swapaxes(screen_buffer, 0, 2)
        screen_buffer = np.swapaxes(screen_buffer, 0, 1)

    # Resize.
    screen_buffer = cv2.resize(screen_buffer, (target_shape[1], target_shape[0]))

    # If it is grayscale, add another dimension.
    if screen_buffer.ndim == 2:
        screen_buffer = np.expand_dims(screen_buffer, axis=2)

    screen_buffer = screen_buffer.astype("float32") / 255.0

    return screen_buffer


if __name__ == "__main__":
    main()
