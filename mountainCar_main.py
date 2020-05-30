from DDQN import Agent
import sys, os, time
import numpy as np
import gym

render = True#draw environment
to_load = True#load from saved model
to_save = True#save model every 50 epochs and at finish

print('Render Environment: ', render)
print('Loading Model: ', to_load)
print('Saving Models: ', to_save)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = env.action_space.n,
                    input_dims = env.observation_space.shape, lr = 1e-4)

    if os.path.isfile('saved_models/trained_model_local.pt') and to_load:
        agent.load_agent()
        agent.epsilon = agent.eps_min

    scores = []
    epochs = 750
    score = 0

    for i in range(epochs):
        if i % 1 == 0 and i > 0:
            avg_score = np.mean(scores[-100:])#last 100 scores to average
            print('epoch: ', i, 'score ', score, 'average score  %.3f' % avg_score,
                    'epsilon %.4f' % agent.epsilon)
        if i % 50 == 0 and i > 0 and to_save:#save the agent every 100 epochs
            agent.save_agent()

        score = 0
        observation = env.reset()
        done = False
        while not done:
            if render:
                env.render()#to draw environment

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
    if to_save:
        agent.save_agent()#save agent after all epochs are done
