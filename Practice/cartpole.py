import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

env = gym.make('CartPole-v1')


#Q = np.zeros([env.observation_space.n, env.action_space.n])

tf.reset_default_graph()

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
n_hidden = int((n_inputs + n_outputs)*2)

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[None,n_inputs],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([n_inputs,n_hidden],0,0.01))
B1 = tf.Variable(tf.random_uniform([n_hidden],0,0.01))
layer1 = tf.nn.relu(tf.matmul(inputs1,W1) + B1)
W2 = tf.Variable(tf.random_uniform([n_hidden,n_outputs],0,0.01))
B2 = tf.Variable(tf.random_uniform([n_outputs],0,0.01))
Qout = tf.matmul(layer1,W2) + B2
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,n_outputs],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout)/32)
trainer = tf.train.RMSPropOptimizer(0.05,0.99,0.0,1e-6)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
experience = []
render = False
successes = 0

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while not d:
            if render:
                env.render()
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.array([s])})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            #print('Q: ' + str(np.max(allQ)))
            s1,r,d,_ = env.step(a[0])
            if j == 475:
                #print('yaay')
                successes = successes + 1
                if successes > 10:
                    render = True
            if j == 600:
                d = True
            experience.append((s,a[0],r,s1,d))

            if (j % 4 == 0 and len(experience) > 32):
                #get random experiences:
                batch_list = random.sample(experience, 32)
                batch = np.array(batch_list)
                nextStates = [list(z[3]) for z in batch_list]
                startStates = [list(z[0]) for z in batch_list]
                actions = [z[1] for z in batch_list]
                rewards = [z[2] for z in batch_list]
                terminals = [z[4] for z in batch_list]
                targets = sess.run(Qout, feed_dict={inputs1:np.array(startStates)})
                Q1 = sess.run(Qout, feed_dict={inputs1:np.array(nextStates)})
                for x in range(32):
                    if terminals[x]:
                        targets[x, actions[x]] = rewards[x]
                    else:
                        targets[x, actions[x]] = rewards[x] + y * np.max(Q1[x])
                _,l = sess.run([updateModel,loss],feed_dict={inputs1:startStates,nextQ:targets})
                #print(l)
            rAll += r
            s = s1
        
        #Reduce chance of random action as we train the model.
        e = 1./((i/50) + 10)
        print('return ' + str(rAll))
        jList.append(j)
        rList.append(rAll)
        
        #fig.show()

print("Average reward: " + str(sum(rList)/num_episodes))

plt.plot(rList)
plt.plot(jList)
plt.show()