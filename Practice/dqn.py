import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import queue
import configparser
import heapq
import scipy.misc
#%matplotlib inline

env_name = 'Breakout-v0'

pause = False
play_mode = False
learn_only_mode = False

learning_rate = 0.00025
momentum = 0.95
decay = 0.95
epsilon = 0.01
error_clipping = True
restore_model = True

y = 0.99
e_initial = 1
e_decay = 0.0006
e_final = 0.1
observe = 10000
target_update_frequency = 5000
update_frequency = 4
action_repeat = 4
replay_memory_size = 100000
no_op_max = 30
batch_size = 32
alpha = 0.5
beta = 0.5

adam = True
double_dqn = True

render = False
render_output_path = "UI/frame.png"
render_repeat = 4

summary_update_frequency = 6
run_no = 38

config_update_frequency = 6

def readConfig():
    global env_name, learning_rate, momentum, decay, epsilon, error_clipping, y, e_initial, e_decay, e_final, observe, target_update_frequency, update_frequency, action_repeat, replay_memory_size, no_op_max, batch_size, beta, alpha, render, render_output_path, render_repeat, summary_update_frequency, run_no, config_update_frequency
    global double_dqn, adam, restore_model, play_mode, pause, learn_only_mode

    config = configparser.ConfigParser()
    config.read('config.ini')

    env_name = config.get('environment', 'env_name')

    play_mode = config.getboolean('app', 'play_mode')
    pause = config.getboolean('app', 'pause')
    learn_only_mode = config.getboolean('app', 'learn_only_mode')

    learning_rate = config.getfloat('optimizer', 'learning_rate')
    momentum = config.getfloat('optimizer', 'momentum')
    decay = config.getfloat('optimizer', 'decay')
    epsilon = config.getfloat('optimizer', 'epsilon')
    error_clipping = config.getboolean('optimizer', 'error_clipping')
    adam = config.getboolean('optimizer', 'adam')
    restore_model = config.getboolean('optimizer', 'restore_model')

    y = config.getfloat('RL', 'y')
    e_initial = config.getfloat('RL', 'e_initial')
    e_decay = config.getfloat('RL', 'e_decay')
    e_final = config.getfloat('RL', 'e_final')
    observe = config.getint('RL', 'observe')
    target_update_frequency = config.getint('RL', 'target_update_frequency')
    update_frequency = config.getint('RL', 'update_frequency')
    action_repeat = config.getint('RL', 'action_repeat')
    replay_memory_size = config.getint('RL', 'replay_memory_size')
    no_op_max = config.getint('RL', 'no_op_max')
    batch_size = config.getint('RL', 'batch_size')
    beta = config.getfloat('RL', 'beta')
    alpha = config.getfloat('RL', 'alpha')
    double_dqn = config.getboolean('RL', 'double_dqn')

    render = config.getboolean('rendering', 'render')
    render_output_path = config.get('rendering', 'render_output_path')
    render_repeat = config.getint('rendering', 'render_repeat')

    summary_update_frequency = config.getint('summary', 'summary_update_frequency')
    run_no = config.getint('summary', 'run_no')

    config_update_frequency = config.getint('config', 'config_update_frequency')

readConfig()


env = gym.make(env_name)

#raw_signal_feed = tf.placeholder(tf.uint8, [210, 160, 3]) 
#raw_signal = tf.Variable(tf.zeros([210, 160, 3], dtype=tf.uint8))
#prev_signal = tf.Variable(tf.zeros([210,160,3], dtype=tf.uint8))
#read_cur_feed = raw_signal.assign(raw_signal_feed)
#copy_cur_to_prev = prev_signal.assign(raw_signal)
#raw_frame = tf.cast(raw_signal, tf.float32)
#prev_frame = tf.cast(prev_signal, tf.float32)
#max_frame = tf.maximum(raw_frame, prev_frame)
#resized_frame = tf.image.resize_images(max_frame, [84, 84])
#final_frame = tf.reduce_mean(resized_frame, axis=2)
#final_frame = tf.cast(final_frame, tf.uint8)

state = tf.placeholder(tf.uint8, [1,84,84,4])
tf.summary.image('state', state)
tf.summary.image('final_frame', tf.transpose(state, perm=[3,1,2,0]), max_outputs=1)


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
        W_fc1_v = weight_variable([3136,512])
        b_fc1_v = bias_variable([512])
        W_fc1_a = weight_variable([3136,512])
        b_fc1_a = bias_variable([512])
        #the final layer for value and advantage respectively:
        W_fc2_v = weight_variable([512,1])
        b_fc2_v = bias_variable([1])
        W_fc2_a = weight_variable([512,n_outputs])
        b_fc2_a = bias_variable([n_outputs])

        input_state_feed = tf.placeholder(tf.uint8, [None, 84, 84, 4], name=name_scope+'input_state_feed')
        input_state_feed_float = tf.cast(input_state_feed, tf.float32)
        #input_state_feed_normalized = input_state_feed_float  - tf.fill([84,84,4], 127.0)

		# hidden layers
        h_conv1 = tf.nn.relu(conv2d(input_state_feed_float,W_conv1,4) + b_conv1)
        #h_conv1_d = tf.nn.dropout(h_conv1, tf.constant(0.66))
        #h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2,2) + b_conv2)
        #h_conv2_d = tf.nn.dropout(h_conv2, tf.constant(0.66))
        h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3)
        #h_conv3_d = tf.nn.dropout(h_conv3, tf.constant(0.66))
        h_conv3_shape = h_conv3.get_shape().as_list()
        #print "dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1_v = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1_v) + b_fc1_v)
        #h_fc1_v_d = tf.nn.dropout(h_fc1_v, tf.constant(0.66))
        h_fc1_a = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1_a) + b_fc1_a)
        #h_fc1_a_d = tf.nn.dropout(h_fc1_a, tf.constant(0.66))

        # Value layer
        V = tf.matmul(h_fc1_v, W_fc2_v) + b_fc2_v

        # Value layer
        A = tf.matmul(h_fc1_a, W_fc2_a) + b_fc2_a

		# Q Value layer
        Q = V + A - tf.reshape(tf.reduce_mean(A, axis=1), [-1,1])
        
        # conv1 = tf.layers.conv2d(input_state_feed, filters = 32, kernel_size = [8,8], strides=(4,4), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation=tf.nn.relu, name=name_scope+'conv1')
        # conv2 = tf.layers.conv2d(conv1, filters = 64, kernel_size = [4,4], strides = (2, 2), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'conv2')
        # conv3 = tf.layers.conv2d(conv2, filters = 64, kernel_size = [3,3], strides = (1,1), kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'conv3')
        # conv3_flat = tf.reshape(conv3, [-1, 7*7*64], name = name_scope+'conv3_flat')
        # full1 = tf.layers.dense(conv3_flat, units = 512, kernel_initializer=weight_intializer, bias_initializer=bias_initializer, activation = tf.nn.relu, name=name_scope+'full1')
        # Q = tf.layers.dense(full1, units = n_outputs, kernel_initializer=weight_intializer, bias_initializer=bias_initializer, name=name_scope+'Q')
        best_action = tf.argmax(Q,1, name=name_scope+'best_action')
        av_action_value = tf.reduce_mean(Q,1,name=name_scope+'av_action_value')
        return input_state_feed, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1_v,b_fc1_v,W_fc1_a,b_fc1_a,W_fc2_v,b_fc2_v,W_fc2_a,b_fc2_a, Q, best_action, av_action_value

#These lines establish the feed-forward part of the network used to choose actions
input_state_feed, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1_v,b_fc1_v,W_fc1_a,b_fc1_a,W_fc2_v,b_fc2_v,W_fc2_a,b_fc2_a, Q, best_action, av_action_value = createBrain('Brain')
tinput_state_feed, tW_conv1,tb_conv1,tW_conv2,tb_conv2,tW_conv3,tb_conv3,tW_fc1_v,tb_fc1_v,tW_fc1_a,tb_fc1_a,tW_fc2_v,tb_fc2_v,tW_fc2_a,tb_fc2_a, tQ, tbest_action, tav_action_value = createBrain('TargetBrain')
copyToTargetBrain = [tW_conv1.assign(W_conv1),tb_conv1.assign(b_conv1),tW_conv2.assign(W_conv2),tb_conv2.assign(b_conv2),tW_conv3.assign(W_conv3),tb_conv3.assign(b_conv3),
                    tW_fc1_v.assign(W_fc1_v),tb_fc1_v.assign(b_fc1_v),tW_fc2_v.assign(W_fc2_v),tb_fc2_v.assign(b_fc2_v),
                    tW_fc1_a.assign(W_fc1_a),tb_fc1_a.assign(b_fc1_a),tW_fc2_a.assign(W_fc2_a),tb_fc2_a.assign(b_fc2_a)]
tf.summary.image('first_conv_layer', tf.transpose(W_conv1, perm=[3,0,1,2]), max_outputs=32)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,n_outputs],dtype=tf.float32)
loss = tf.reduce_mean(tf.square(nextQ - Q))
loss_feed = tf.placeholder(tf.float32)
tf.summary.scalar('loss', loss_feed)
if adam:
    trainer = tf.train.AdamOptimizer(learning_rate)
else:
    trainer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
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

# a min heap of experiences based on selection probability
priority_list = []
expid_to_heap_index_map = {}
equiprobable_buckets = []

priority_list_feed = tf.placeholder(tf.float32, [None])
tf.summary.histogram('priority_list', priority_list_feed)

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

def calculateEquiprobableBuckets(alpha, N, k):
    # assume there are k sets of experiences. Each of size N/k. In ith set, each experience has probability (1/i)^alpha
    # we need to compute k buckets each with cumulative probability sum_of_probabilities/k
    buckets = []
    sum_of_probabilities = 0
    for set_no in range(k):
        sum_of_probabilities += N * pow(.1/float(set_no+1),alpha) / k
    
    # print(sum_of_probabilities)

    i = 0
    sum_so_far = 0
    while (len(buckets) < k-1):
        while (sum_so_far < (len(buckets) + 1) * sum_of_probabilities/k):
            set_no = (k * i) / N
            sum_so_far += pow(.1/float(k-set_no), alpha)
            i+=1
        buckets.append(i)

    buckets.append(N)

    #print(buckets)
    return buckets

def update_expid_to_heap_index_map(heap_index):
    expid = priority_list[heap_index][1]
    expid_to_heap_index_map[expid] = heap_index

def compare(err_exp_tuple1, err_exp_tuple2):
    if err_exp_tuple1[0] < err_exp_tuple2[0]: return -1
    if err_exp_tuple1[0] > err_exp_tuple2[0]: return 1
    else:
        if err_exp_tuple1[1] < err_exp_tuple2[1]: return -1
        if err_exp_tuple1[1] > err_exp_tuple2[1]: return 1
        return 0

def sort_min_heap():
    priority_list.sort()
    for heap_index in range(len(priority_list)): update_expid_to_heap_index_map(heap_index)

def balance_min_heap(index):
    if index == 0: return

    parent_index = int((index - 1)/2)
    if compare(priority_list[index], priority_list[parent_index]) < 0:
        # need to push index up:
        priority_list[index], priority_list[parent_index] = priority_list[parent_index], priority_list[index]
        update_expid_to_heap_index_map(index)
        update_expid_to_heap_index_map(parent_index)
        # maybe need to push further up?
        balance_min_heap(parent_index)
    else:
        #check if this guy is larger than any of the children:
        l_child_index = 2 * index + 1
        r_child_index = 2 * index + 2
        if l_child_index < len(priority_list) and compare(priority_list[index], priority_list[l_child_index]) > 0:
            priority_list[index], priority_list[l_child_index] = priority_list[l_child_index], priority_list[index]
            update_expid_to_heap_index_map(index)
            update_expid_to_heap_index_map(l_child_index)
            balance_min_heap(l_child_index)
        elif r_child_index < len(priority_list) and compare(priority_list[index], priority_list[r_child_index]) > 0:
            priority_list[index], priority_list[r_child_index] = priority_list[r_child_index], priority_list[index]
            update_expid_to_heap_index_map(index)
            update_expid_to_heap_index_map(r_child_index)
            balance_min_heap(r_child_index)

def setPrioriry(expid, rank):
    priority = 1./float(rank)
    selectionProbability = pow(priority, alpha)
    record = [selectionProbability, expid]
    if expid < len(priority_list):
        heap_index = expid_to_heap_index_map[expid]
        priority_list[heap_index] = record
    else:
        # the case when replay buff is not full yet. in this case, add at the end of heap:
        priority_list.append(record)
        heap_index = len(priority_list) - 1
        update_expid_to_heap_index_map(heap_index)
    
    #now balance the heap:
    balance_min_heap(heap_index)


def addToExperience(startState, action, nextState, reward, terminal, rank):
    global experience_count
    experience_startStates[experience_count % replay_memory_size] = startState
    experience_actions[experience_count % replay_memory_size] = action
    experience_nextStates[experience_count % replay_memory_size] = nextState
    experience_rewards[experience_count % replay_memory_size] = reward
    experience_terminals[experience_count % replay_memory_size] = terminal

    # add it to priority list:
    setPrioriry(experience_count % replay_memory_size, rank)

    # every replay_memory_size interval, sort the priority_list:
    if experience_count % replay_memory_size == 0:
        sort_min_heap()

    experience_count += 1

    global equiprobable_buckets
    if experience_count > 0 and experience_count <= replay_memory_size and experience_count % 2000 == 0:
        equiprobable_buckets = calculateEquiprobableBuckets(alpha, experience_count, batch_size)
    
    #return where it was added:
    return (experience_count - 1) % replay_memory_size

def selectExperiencesMinibatch():
    indices = []

    start_index = 0
    for end_index in equiprobable_buckets:
        heap_index = random.randint(start_index, end_index - 1)
        index = priority_list[heap_index][1]
        indices.append(index)
        start_index = end_index
   
    return indices, experience_startStates[indices], experience_actions[indices], experience_nextStates[indices], experience_rewards[indices], experience_terminals[indices]

#prev_raw_signal = np.zeros((210, 160, 3), "uint8")

def getCurFrame(cur_raw_signal, sess):
    #sess.run(copy_cur_to_prev)
    #sess.run(read_cur_feed, feed_dict = {raw_signal_feed: cur_raw_signal})
    #raw_signal_feed = tf.placeholder(tf.uint8, [210, 160, 3]) 
    #raw_signal = tf.Variable(tf.zeros([210, 160, 3], dtype=tf.uint8))
    #prev_signal = tf.Variable(tf.zeros([210,160,3], dtype=tf.uint8))
    #read_cur_feed = raw_signal.assign(raw_signal_feed)
    #copy_cur_to_prev = prev_signal.assign(raw_signal)
    #raw_frame = tf.cast(raw_signal, tf.float32)
    #prev_frame = tf.cast(prev_signal, tf.float32)
    #max_frame = tf.maximum(raw_frame, prev_frame)
    #resized_frame = tf.image.resize_images(max_frame, [84, 84])
    #final_frame = tf.reduce_mean(resized_frame, axis=2)
    #final_frame = tf.cast(final_frame, tf.uint8)
    #global prev_raw_signal
    frame = np.mean(cur_raw_signal, 2).astype("uint8")
    frame = scipy.misc.imresize(frame, [84, 84])
    frame = frame.reshape([84,84])
    #prev_raw_signal = cur_raw_signal
    #return sess.run(final_frame)
    return frame

def getCurState(cur_raw_signal, sess):
    curFrame = getCurFrame(cur_raw_signal, sess)
    last_m_frames.append(curFrame)
    if len(last_m_frames) > m:
        last_m_frames.pop(0)
    return np.array(last_m_frames).transpose(1,2,0).reshape([1,84,84,m])

def chooseAction(s, sess):
    #choose an action
    if not play_mode and f < observe:
        return env.action_space.sample()

    global e
    if e > e_final:
        e = e / e_decay
    else:
        e = e_final
    if (np.random.rand(1) > e or play_mode):
        global current_av_action_value
        a, current_av_action_value = sess.run([best_action, av_action_value], feed_dict={input_state_feed:s})
        a = a[0]
        current_av_action_value = current_av_action_value[0]
    else:
        a = env.action_space.sample()
    return a


init = tf.initialize_all_variables()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/run' + str(run_no) +  'per_dualing_double_dqn_ec_' + '_adam-' + str(adam) + '_' + str(env_name) + "_" + str(learning_rate) + '_' + str(decay) + '_' + str(momentum) + '_' + str(epsilon))
saver = tf.train.Saver()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(config=config) as sess:
    writer.add_graph(sess.graph)
    sess.run(init)
    sess.run(copyToTargetBrain)
    if restore_model:
        try:
            saver.restore(sess, "ckpts/model_" + env_name + ".ckpt")
        except Exception:
            print('Could not restore model')
    ep_no = 0
    rAll = 0
    while True:
        # Reset environment and get first new observation
        obs = env.reset()
        # Let go off a random number of frames without any action so that agent faces a random start
        for i in range(random.randint(0,no_op_max)):
            obs,r,d,_ = env.step(0)
        s = getCurState(obs, sess)
        a = env.action_space.sample()
        lastrAll = rAll
        rAll = 0
        d = False
        j = 0
        while not d:
            if render:
                # if (f % render_repeat == 0): env.render()
                if (f % render_repeat == 0): mpimg.imsave(render_output_path, obs)
            
            #choose action:
            if j % action_repeat == (action_repeat - 1):
                a = chooseAction(s, sess)
            #print(a)
            
            #take action and collect reward and see next state
            obs,r,d,_ = env.step(a)
            s1 = getCurState(obs, sess)

            #store the experience
            if abs(r) > 0: rank = 1
            else: rank = 2
            exp_id = addToExperience(s, a, s1, r, d, rank)

            #do learning here
            if (not play_mode and f > observe and f % update_frequency == 0 and experience_count > batch_size):
                #get random experiences:
                exp_ids, startStates, actions, nextStates, rewards, terminals = selectExperiencesMinibatch()
                targets = sess.run(Q, feed_dict={input_state_feed:startStates})
                if double_dqn:
                    next_best_actions = sess.run(best_action, feed_dict={input_state_feed:nextStates})
                Q1 = sess.run(tQ, feed_dict={tinput_state_feed:nextStates})
                exp_err_tuples = [0 for i in range(batch_size)]
                for x in range(batch_size):
                    if terminals[x]:
                        err = rewards[x] - targets[x, actions[x]]
                    else:
                        if double_dqn:
                            err = rewards[x] + y * Q1[x, next_best_actions[x]] - targets[x, actions[x]]
                        else:
                            err = rewards[x] + y * np.max(Q1[x])- targets[x, actions[x]]
                    exp_err_tuples[x] = [abs(err), exp_ids[x]]
                    current_error += 0.5 * (err - current_error)
                    # to stabilize training:
                    if error_clipping:
                        if err > 1: err = 1
                        if err < -1: err = -1
                    targets[x, actions[x]] += err
                exp_err_tuples.sort()
                for x in range(batch_size): setPrioriry(exp_err_tuples[x][1], batch_size - x)
                _,current_loss = sess.run([updateModel,loss],feed_dict={input_state_feed:startStates,nextQ:targets})

            if (not play_mode and f % target_update_frequency == 0):
                sess.run(copyToTargetBrain)

            if f % summary_update_frequency == 0:
                writer.add_summary(sess.run(merged, feed_dict={state:s, 
                                                    totalReward_feed:totalReward,
                                                    loss_feed:current_loss,
                                                    epsilon_feed:e,
                                                    err_feed:current_error,
                                                    av_action_value_feed:current_av_action_value,
                                                    rAll_feed: lastrAll,
                                                    priority_list_feed: np.array(priority_list)[:,0]}),
                                                    f)
            
            if f % 50 * config_update_frequency == 0:
                saver.save(sess, "ckpts/model_" + env_name + ".ckpt")

            if f % config_update_frequency == 0:
                readConfig()

            f = f + 1
            j = j + 1
            rAll += r
            totalReward += r
            s = s1
        ep_no += 1
        print('Episode : ' + str(ep_no) + '\tReward: ' + str(rAll))