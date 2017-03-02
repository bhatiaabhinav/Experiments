import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import queue
import configparser
import heapq
#%matplotlib inline

env_name = 'Breakout-v0'

learning_rate = 0.00025
momentum = 0.95
decay = 0.95
epsilon = 0.01
error_clipping = True

y = 0.99
e_initial = 1
e_decay = 0.0006
e_final = 0.1
observe = 10000
target_update_frequency = 5000
update_frequency = 4
action_repeat = 4
replay_memory_size = 100000
batch_size = 32
reserved_list_max_size = 2
priority_list_max_size = 16

render = False
render_output_path = "UI/frame.png"
render_repeat = 4

summary_update_frequency = 6
run_no = 38

config_update_frequency = 6

def readConfig():
    global env_name, learning_rate, momentum, decay, epsilon, error_clipping, y, e_initial, e_decay, e_final, observe, target_update_frequency, update_frequency, action_repeat, replay_memory_size, batch_size, reserved_list_max_size, priority_list_max_size, render, render_output_path, render_repeat, summary_update_frequency, run_no, config_update_frequency

    config = configparser.ConfigParser()
    config.read('config.ini')

    env_name = config.get('environment', 'env_name')

    learning_rate = config.getfloat('optimizer', 'learning_rate')
    momentum = config.getfloat('optimizer', 'momentum')
    decay = config.getfloat('optimizer', 'decay')
    epsilon = config.getfloat('optimizer', 'epsilon')
    error_clipping = config.getboolean('optimizer', 'error_clipping')

    y = config.getfloat('RL', 'y')
    e_initial = config.getfloat('RL', 'e_initial')
    e_decay = config.getfloat('RL', 'e_decay')
    e_final = config.getfloat('RL', 'e_final')
    observe = config.getint('RL', 'observe')
    target_update_frequency = config.getint('RL', 'target_update_frequency')
    update_frequency = config.getint('RL', 'update_frequency')
    action_repeat = config.getint('RL', 'action_repeat')
    replay_memory_size = config.getint('RL', 'replay_memory_size')
    batch_size = config.getint('RL', 'batch_size')
    reserved_list_max_size = config.getint('RL', 'reserved_list_max_size')
    priority_list_max_size = config.getint('RL', 'priority_list_max_size')

    render = config.getboolean('rendering', 'render')
    render_output_path = config.get('rendering', 'render_output_path')
    render_repeat = config.getint('rendering', 'render_repeat')

    summary_update_frequency = config.getint('summary', 'summary_update_frequency')
    run_no = config.getint('summary', 'run_no')

    config_update_frequency = config.getint('config', 'config_update_frequency')

readConfig()


env = gym.make(env_name)

raw_signal_feed = tf.placeholder(tf.uint8, [210, 160, 3]) 
raw_signal = tf.Variable(tf.zeros([210, 160, 3], dtype=tf.uint8))
prev_signal = tf.Variable(tf.zeros([210,160,3], dtype=tf.uint8))
read_cur_feed = raw_signal.assign(raw_signal_feed)
copy_cur_to_prev = prev_signal.assign(raw_signal)
raw_frame = tf.cast(raw_signal, tf.float32)
prev_frame = tf.cast(prev_signal, tf.float32)
max_frame = tf.maximum(raw_frame, prev_frame)
resized_frame = tf.image.resize_images(max_frame, [84, 84])
final_frame = tf.reduce_mean(resized_frame, axis=2)
final_frame = tf.cast(final_frame, tf.uint8)
tf.summary.image('final_frame', tf.reshape(final_frame, [1, 84, 84, 1]))

state = tf.placeholder(tf.uint8, [1,84,84,4])
tf.summary.image('state', state)


n_outputs = env.action_space.n
print(n_outputs)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def createBrain(name_scope):
    with tf.name_scope(name_scope):
        W_conv1 = weight_variable([8,8,4,32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([4,4,32,64])
        b_conv2 = bias_variable([64])
        W_conv3 = weight_variable([3,3,64,64])
        b_conv3 = bias_variable([64])
        W_fc1 = weight_variable([3136,512])
        b_fc1 = bias_variable([512])
        W_fc2 = weight_variable([512,n_outputs])
        b_fc2 = bias_variable([n_outputs])

        input_state_feed = tf.placeholder(tf.uint8, [None, 84, 84, 4], name=name_scope+'input_state_feed')
        input_state_feed_float = tf.cast(input_state_feed, tf.float32)

		# hidden layers
        h_conv1 = tf.nn.relu(conv2d(input_state_feed_float,W_conv1,4) + b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2,2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        #print "dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

		# Q Value layer
        Q = tf.matmul(h_fc1,W_fc2) + b_fc2
        
        # conv1 = tf.layers.conv2d(input_state_feed, filters = 32, kernel_size = [8,8], strides=(4,4), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation=tf.nn.relu, name=name_scope+'conv1')
        # conv2 = tf.layers.conv2d(conv1, filters = 64, kernel_size = [4,4], strides = (2, 2), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'conv2')
        # conv3 = tf.layers.conv2d(conv2, filters = 64, kernel_size = [3,3], strides = (1,1), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'conv3')
        # conv3_flat = tf.reshape(conv3, [-1, 7*7*64], name = name_scope+'conv3_flat')
        # full1 = tf.layers.dense(conv3_flat, units = 512, kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'full1')
        # Q = tf.layers.dense(full1, units = n_outputs, kernel_initializer=weight_intializer, bias_initializer=bias_initializer, name=name_scope+'Q')
        best_action = tf.argmax(Q,1, name=name_scope+'best_action')
        av_action_value = tf.reduce_mean(Q,1,name=name_scope+'av_action_value')
        return input_state_feed, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2, Q, best_action, av_action_value

#These lines establish the feed-forward part of the network used to choose actions
input_state_feed, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2, Q, best_action, av_action_value = createBrain('Brain')
tinput_state_feed, tW_conv1,tb_conv1,tW_conv2,tb_conv2,tW_conv3,tb_conv3,tW_fc1,tb_fc1,tW_fc2,tb_fc2, tQ, tbest_action, tav_action_value = createBrain('TargetBrain')
copyToTargetBrain = [tW_conv1.assign(W_conv1),tb_conv1.assign(b_conv1),tW_conv2.assign(W_conv2),tb_conv2.assign(b_conv2),tW_conv3.assign(W_conv3),tb_conv3.assign(b_conv3),tW_fc1.assign(W_fc1),tb_fc1.assign(b_fc1),tW_fc2.assign(W_fc2),tb_fc2.assign(b_fc2)]
tf.summary.image('first_conv_layer', tf.transpose(W_conv1, perm=[3,0,1,2]), max_outputs=32)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,n_outputs],dtype=tf.float32)
loss = tf.reduce_mean(tf.square(nextQ - Q))
loss_feed = tf.placeholder(tf.float32)
tf.summary.scalar('loss', loss_feed)
#trainer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
trainer = tf.train.AdamOptimizer(learning_rate)
updateModel = trainer.minimize(loss)





epsilon_feed = tf.placeholder(tf.float32)
tf.summary.scalar('epsilon', epsilon_feed)
#create lists to contain total rewards and steps per episode
jList = []
rList = []
e = e_initial
totalReward = 0
totalReward_feed = tf.placeholder(tf.float32)
tf.summary.scalar('totalReward', totalReward_feed)
experience_startStates = np.zeros(replay_memory_size * 84 * 84 * 4, dtype='uint8').reshape([replay_memory_size,84,84,4])
experience_nextStates = np.zeros(replay_memory_size * 84 * 84 * 4, dtype='uint8').reshape([replay_memory_size,84,84,4])
experience_actions = np.zeros(replay_memory_size, dtype='int32')
experience_rewards = np.zeros(replay_memory_size)
experience_terminals = np.zeros(replay_memory_size, dtype='bool')
experience_count = 0

priority_experiences = []
reserved_list = []

successes = 0
f = 0
m = 4
current_loss = 0
last_m_frames = [np.zeros([84,84],dtype='uint8') for i in range(m)]
current_av_action_value = 0
av_action_value_feed = tf.placeholder(tf.float32)
tf.summary.scalar('av_action_value', av_action_value_feed)
err_feed = tf.placeholder(tf.float32)
tf.summary.scalar('error', err_feed)
current_error = 0
rAll_feed = tf.placeholder(tf.int32)
tf.summary.scalar('reward_per_episode', rAll_feed)

def addToExperience(startState, action, nextState, reward, terminal, reserved):
    global experience_count
    experience_startStates[experience_count % replay_memory_size] = startState
    experience_actions[experience_count % replay_memory_size] = action
    experience_nextStates[experience_count % replay_memory_size] = nextState
    experience_rewards[experience_count % replay_memory_size] = reward
    experience_terminals[experience_count % replay_memory_size] = terminal

    if (reserved_list_max_size > 0 and reserved):
        reserved_list.append(experience_count % replay_memory_size)
        if(len(reserved_list) > reserved_list_max_size):
            reserved_list.pop(0)

    experience_count += 1
    #return where it was added:
    return (experience_count - 1) % replay_memory_size

def getRandomBatchFromExperience():
    count = experience_count
    if count > replay_memory_size:
        count = replay_memory_size
    indices = []
    # add all experiences from reserved list:
    indices.extend(reserved_list)
    reserved_list.clear()
    #1 seat reserved for the latest experience:
    #indices.append((experience_count - 1) % replay_memory_size)
    if len(priority_experiences) > 0:
        for i in range(len(priority_experiences)):
            indices.append(priority_experiences[i][2])
        priority_experiences.clear()
    indices.extend(np.random.choice(count, batch_size - len(indices), replace=False))
    return indices, experience_startStates[indices], experience_actions[indices], experience_nextStates[indices], experience_rewards[indices], experience_terminals[indices]

pq_counter = 0
def ifNeededAddToPriorityExperiences(exp_id, exp_error):
    global pq_counter
    if (priority_list_max_size == 0):
        return
    if (len(priority_experiences) < priority_list_max_size):
        heapq.heappush(priority_experiences, (abs(exp_error), pq_counter, exp_id))
        pq_counter += 1
    else:
        if abs(exp_error) > priority_experiences[0][-1]:
            heapq.heappushpop(priority_experiences, (abs(exp_error), pq_counter, exp_id))
            pq_counter += 1

def getCurFrame(cur_raw_signal, sess):
    sess.run(copy_cur_to_prev)
    sess.run(read_cur_feed, feed_dict = {raw_signal_feed: cur_raw_signal})
    return sess.run(final_frame)

def getCurState(cur_raw_signal, sess):
    curFrame = getCurFrame(cur_raw_signal, sess)
    last_m_frames.append(curFrame)
    if len(last_m_frames) > m:
        last_m_frames.pop(0)
    return np.array(last_m_frames).transpose(1,2,0).reshape([1,84,84,m])

def chooseAction(s, sess):
    #choose an action
    if f < observe:
        return env.action_space.sample()

    global e
    if e > e_final:
        e = e / e_decay
    else:
        e = e_final
    if (np.random.rand(1) > e):
        global current_av_action_value
        a, current_av_action_value = sess.run([best_action, av_action_value], feed_dict={input_state_feed:s})
        a = a[0]
        current_av_action_value = current_av_action_value[0]
    else:
        a = env.action_space.sample()
    return a


init = tf.initialize_all_variables()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/run' + str(run_no) +  '_adam_dqn_' + str(env_name) + "_" + str(learning_rate) + '_' + str(decay) + '_' + str(momentum) + '_' + str(epsilon))

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(init)
    sess.run(copyToTargetBrain)
    ep_no = 0
    while True:
        #Reset environment and get first new observation
        obs = env.reset()
        s = getCurState(obs, sess)
        a = env.action_space.sample()
        rAll = 0
        d = False
        j = 0
        while not d:
            if render:
                #env.render()
                if (f % render_repeat == 0): mpimg.imsave(render_output_path, obs)
            
            #choose action:
            if j % action_repeat == (action_repeat - 1):
                a = chooseAction(s, sess)
            #print(a)
            
            #take action and collect reward and see next state
            obs,r,d,_ = env.step(a)
            s1 = getCurState(obs, sess)

            #store the experience
            exp_id = addToExperience(s, a, s1, r, d, True)

            #do learning here
            if (f > observe and f % update_frequency == 0 and experience_count > batch_size):
                #get random experiences:
                # batch_list = random.sample(experience, batch_size)
                # batch = np.array(batch_list)
                # nextStates = np.array([list(z[3][0]) for z in batch_list])
                # startStates = np.array([list(z[0][0]) for z in batch_list])
                # actions = [z[1] for z in batch_list]
                # rewards = [z[2] for z in batch_list]
                # terminals = [z[4] for z in batch_list]
                exp_ids, startStates, actions, nextStates, rewards, terminals = getRandomBatchFromExperience()
                targets = sess.run(Q, feed_dict={input_state_feed:startStates})
                #next_best_actions = sess.run(best_action, feed_dict={input_state_feed:nextStates})
                Q1 = sess.run(tQ, feed_dict={tinput_state_feed:nextStates})
                for x in range(batch_size):
                    if terminals[x]:
                        err = rewards[x] - targets[x, actions[x]]
                    else:
                        err = rewards[x] + y * np.max(Q1[x])- targets[x, actions[x]]
                    ifNeededAddToPriorityExperiences((exp_ids[x] - 1) % replay_memory_size, err)
                    current_error += 0.5 * (err - current_error)
                    # to stabilize training:
                    if error_clipping:
                        if err > 1: err = 1
                        if err < -1: err = -1
                    targets[x, actions[x]] += err

                _,current_loss = sess.run([updateModel,loss],feed_dict={input_state_feed:startStates,nextQ:targets})

            if (f % target_update_frequency == 0):
                sess.run(copyToTargetBrain)

            if f % summary_update_frequency == 0:
                writer.add_summary(sess.run(merged, feed_dict={state:s, 
                                                    totalReward_feed:totalReward,
                                                    loss_feed:current_loss,
                                                    epsilon_feed:e,
                                                    err_feed:current_error,
                                                    av_action_value_feed:current_av_action_value,
                                                    rAll_feed: rAll}),
                                                    f)

            if f % config_update_frequency == 0:
                readConfig()

            f = f + 1
            j = j + 1
            rAll += r
            totalReward += r
            s = s1
        ep_no += 1
        print('Episode : ' + str(ep_no) + '\tReward: ' + str(rAll))