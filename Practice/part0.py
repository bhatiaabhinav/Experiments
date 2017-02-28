import gym
import numpy as np

env = gym.make('FrozenLake-v0')


Q = np.zeros([env.observation_space.n, env.action_space.n])

gamma = 0.99
alpha = 0.85
N = 2000

RList = []
for n in range(N) :
    epsilon = 1./(1 + n)
    s = env.reset()
    R = 0
    done = False


    while not done:


        action = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n) * epsilon)
        #collect reward and observe new state
        ns, reward, done, _ = env.step(action)
        #update based on reward and new state
        Q[s,action] += alpha * (reward + gamma * np.max(Q[ns, :]) - Q[s,action])
        R += reward
        s = ns

    RList.append(R)

    print('return: ' + str(R))

print("Average Return: " +  str(sum(RList)/N))
print("Final Q-Table Values")
print(Q)