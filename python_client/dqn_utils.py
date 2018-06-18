import time, sys
import numpy as np
import random

import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from PIL import Image

import comm

import pdb

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    현재 가지고 있는 q-value를 가지고 policy를 결정(어떤 state에서 무슨 action을 얼마만큼의 확률로 할지를 결정)

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    # pdb.set_trace()
    def policy_fn(sess, observation, epsilon):
        angle_A = np.ones(nA[0], dtype=float) * epsilon / nA[0]
        taptime_A = np.ones(nA[1], dtype=float) * epsilon / nA[1]
        # A = np.ones(nA, dtype=float) * epsilon / nA # action 수 만큼의 길이

        # pdb.set_trace()
        [angle_q_values, taptime_q_values] = estimator.predict(sess, np.expand_dims(observation, 0))
        # q_values = estimator.predict(sess, observation)[0]
        # pdb.set_trace()
        angle_best_action = np.argmax(angle_q_values)
        taptime_best_action = np.argmax(taptime_q_values)
        angle_A[angle_best_action] += (1.0 - epsilon)
        taptime_A[taptime_best_action] += (1.0 - epsilon)
        return angle_A, taptime_A
    return policy_fn

def make_epsilon_greedy_policy_parNN(estimator, nA, soft=False):

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA # action 수 만큼의 길이
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        # print(q_values)
        if soft:
            return q_values
        else:
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            # print(q_values, best_action)
            return q_values, A # 원래는 A만 있었음
    return policy_fn

def get_feature_4096(model, img_path, need_crop = True, need_resize = True):
    '''
    vgg16 = VGG16(weights= 'imagenet', include_top= False) # output: 7*7*512
    vgg16 = VGG16(weights= 'imagenet', include_top= True) # output: 1000*1

    Return:
      4096*1
    '''
    model_4096 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    im = Image.open(img_path)

    if need_crop == True:
        im = im.crop((70,45,760,390))
    if need_resize == True:
        im = im.resize((224,224))

    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # pdb.set_trace()

    return model_4096.predict(x).reshape(4096)

def copy_model_parameters(sess, estimator_from, estimator_to):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator_from: Estimator to copy the paramters from
      estimator_to: Estimator to copy the parameters to
    """
    # pdb.set_trace()
    # pdb.set_trace()
    e_from_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_from.scope)]
    e_from_params = sorted(e_from_params, key=lambda v: v.name)
    e_to_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_to.scope)]
    e_to_params = sorted(e_to_params, key=lambda v: v.name)

    update_ops = []
    for e_from_v, e_to_v in zip(e_from_params, e_to_params):
        op = e_to_v.assign(e_from_v)
        update_ops.append(op)

    sess.run(update_ops)

def get_slingshot_refpoint(slingshot):
	X_OFFSET = 0.5
	Y_OFFSET = 0.65
	x = slingshot[0]+slingshot[2]*X_OFFSET
	y = slingshot[1]+slingshot[2]*Y_OFFSET
	return (x,y)

def get_score_after_shot(current_dir, parser, comm_socket, start_score,fast = False):
    """
    Get score after shot.

    Args:
      parser : Wrapper object
      comm_socket : server-client socket
      start_score : score before shooting
    Return:
      reward (last_score - start_score)
      new_score (last_score)
      save_path (next state raw image path. WON or LOST should be captured at terminal state.)
      state (PLAYING or WON or LOST)
    """

    end_image = None
    save_path = None
    screenshot = None
    last_score = start_score
    sleepcount = 0
    if fast:
        full_sleep = 6
    else:
        full_sleep = 15

    # Shooting is done. Check the score
    while True:
        # check the status of the screenshot
        save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
        screenshot = comm.comm_do_screenshot(comm_socket, save_path=save_path)
        if comm.comm_get_state(comm_socket) == 'WON':
            score = parser.get_score_end_game(save_path)
        else:
            score  = parser.get_score_in_game(save_path)

        # if the new score is less than the last score, something went wrong, just return the last status
        # pdb.set_trace()

        if (score > last_score):
            end_image = screenshot
            last_score = score
            sleepcount = 0

        # PLAYING / WON / LOST
        elif last_score==score:
            if (comm.comm_get_state(comm_socket) == 'LOST'):
                save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
                end_image = comm.comm_do_screenshot(comm_socket, save_path=save_path)
                break
            if comm.comm_get_state(comm_socket) == 'WON':
                time.sleep(1)
                save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
                end_image = comm.comm_do_screenshot(comm_socket, save_path=save_path)
                last_score = parser.get_score_end_game(save_path)
                break
            time.sleep(1)
            sleepcount+=1
            if sleepcount>=full_sleep:
                save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
                end_image = comm.comm_do_screenshot(comm_socket, save_path=save_path)
                print('over slept')
                sys.stdout.flush()
                break

        else: # last_score > score:
            if comm.comm_get_state(comm_socket) == 'WON' or comm.comm_get_state(comm_socket) == 'LOST':
                save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
                end_image = comm.comm_do_screenshot(comm_socket, save_path=save_path)
                if comm.comm_get_state(comm_socket) == 'WON':
                    last_score = parser.get_score_end_game(save_path)
                else:
                    break
            else:
                print("something wrong...")

        # if the no change count and passed time are enough, return.
    state = comm.comm_get_state(comm_socket)
    return last_score-start_score, last_score, save_path, state

def clear_screenshot(path):
    import os, glob
    l = glob.glob(os.path.join(path, '*'))
    for f in l:
        os.remove(f)

def init_replaymemory_all_levels(exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    i = 0
    for filename in glob.iglob("%s/*/*/s_?_*.png_seg.png"%(exp_path)):
        i += 1
        if i%300 == 0:
            print(i)
        level_onehot = np.zeros(21, dtype = int)
        for j in range(21):
            if "level{}_".format(j+1) in filename:
                level_onehot[j] = 1
                break

        next_state = os.path.abspath(filename)
        if "PLAYING" in next_state:
            game_state = "PLAYING"
        elif "LOST" in next_state:
            game_state = "LOST"
        elif "WON" in next_state:
            game_state = "WON"
        else:
            game_state = None
            print("game_state error: None")
        dir = os.path.dirname(next_state)
        state = glob.glob("%s/s_?.png_seg.png"%dir)
        action, reward= None, None
        with open(os.path.join(dir, 'action'), 'rb') as f:
            action = pickle.load(f)
        with open(os.path.join(dir, 'reward'), 'rb') as f:
            reward = pickle.load(f)
        state = np.concatenate((level_onehot, get_feature_4096(model=model_name, img_path=state[0])))
        next_state = np.concatenate((level_onehot, get_feature_4096(model=model_name, img_path=next_state)))
        replay_memory.append([state, action, reward, next_state, game_state])
    return replay_memory

def init_replaymemory_all(exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    i = 0
    for filename in glob.iglob("%s/*/*/s_?_*.png_seg.png"%(exp_path)):
        i += 1
        if i%300 == 0:
            print(i)

        next_state = os.path.abspath(filename)
        if "PLAYING" in next_state:
            game_state = "PLAYING"
        elif "LOST" in next_state:
            game_state = "LOST"
        elif "WON" in next_state:
            game_state = "WON"
        else:
            game_state = None
            print("game_state error: None")
        dir = os.path.dirname(next_state)
        state = glob.glob("%s/s_?.png_seg.png"%dir)
        action, reward= None, None
        with open(os.path.join(dir, 'action'), 'rb') as f:
            action = pickle.load(f)
        with open(os.path.join(dir, 'reward'), 'rb') as f:
            reward = pickle.load(f)
        state = get_feature_4096(model=model_name, img_path=state[0])
        next_state = get_feature_4096(model=model_name, img_path=next_state)
        replay_memory.append([state, action, reward, next_state, game_state])
    return replay_memory

def init_replaymemory(angle_step, exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    if angle_step>1:
        angles = [i for i in range(5,90) if (i%angle_step==0)]
    else:
        angles = range(5,86)
    for angle in angles:
        for filename in glob.iglob("%s/*/*/s_?_%d_*_seg.png"%(exp_path, angle)):
            next_state = os.path.abspath(filename)
            if "PLAYING" in next_state:
                game_state = "PLAYING"
            elif "LOST" in next_state:
                game_state = "LOST"
            elif "WON" in next_state:
                game_state = "WON"
            else:
                game_state = None
                print("game_state error: None")
            dir = os.path.dirname(next_state)
            state = glob.glob("%s/s_?.png_seg.png"%dir)
            action, reward= None, None
            with open(os.path.join(dir, 'action'), 'rb') as f:
                action = pickle.load(f)
            with open(os.path.join(dir, 'reward'), 'rb') as f:
                reward = pickle.load(f)
            state = get_feature_4096(model=model_name, img_path=state[0])
            next_state = get_feature_4096(model=model_name, img_path=next_state)
            replay_memory.append([state, action, reward, next_state, game_state])
    return replay_memory

def init_replaymemory_WithAngleSet(angle_set, exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    for angle in angle_set:
        print(angle)
        for filename in glob.iglob("%s/*/*/s_?_%d_*_seg.png"%(exp_path, angle)):
            next_state = os.path.abspath(filename)
            if "PLAYING" in next_state:
                game_state = "PLAYING"
            elif "LOST" in next_state:
                game_state = "LOST"
            elif "WON" in next_state:
                game_state = "WON"
            else:
                game_state = None
                print("game_state error: None")
            dir = os.path.dirname(next_state)
            state = glob.glob("%s/s_?.png_seg.png"%dir)
            action, reward= None, None
            with open(os.path.join(dir, 'action'), 'rb') as f:
                action = pickle.load(f)
            with open(os.path.join(dir, 'reward'), 'rb') as f:
                reward = pickle.load(f)
            state = get_feature_4096(model=model_name, img_path=state[0])
            next_state = get_feature_4096(model=model_name, img_path=next_state)
            replay_memory.append([state, action, reward, next_state, game_state])
    return replay_memory

def pretrain(replay_memory, valid_angles, valid_taptimes, estimator, target_estimator, sess, batch_size = 6, discount_factor = 0.7):
    samples = random.sample(replay_memory, batch_size)
    states_batch, action_batch, reward_batch, next_states_batch, game_state_batch = map(np.array, zip(*samples))
    reward_batch = np.clip(reward_batch/10000, 0, 6)
    # (1,1,4096) (1,2) (1,) (1,)

    done_batch = np.array([1 if (game_state =='LOST' or 'WON') else 0 for game_state in game_state_batch])

    # angle_action_batch = np.array([action_batch[i][0] for i in range(batch_size)])
    # taptime_action_batch = np.array([action_batch[i][1] for i in range(batch_size)])

    angle_action_batch_idx = np.array([valid_angles.index(action_batch[i][0]) for i in range(batch_size)])
    taptime_action_batch_idx = np.array([valid_taptimes.index(action_batch[i][1]) for i in range(batch_size)])

    # 학습에 넣을 target reward 계산

    [angle_q_values_next, taptime_q_values_next] = estimator.predict(sess, next_states_batch)
    best_angle_actions = np.argmax(angle_q_values_next, axis=1)
    # taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
    best_taptime_actions = np.argmax(taptime_q_values_next, axis=1)

    [angle_q_values_next_target, taptime_q_values_next_target] = estimator.predict(sess, next_states_batch)
    # taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)

    angle_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        discount_factor * angle_q_values_next_target[np.arange(batch_size), best_angle_actions]

    taptime_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        discount_factor * taptime_q_values_next_target[np.arange(batch_size), best_taptime_actions]

    # Perform gradient descent update
    states_batch = np.array(states_batch)
    loss = estimator.update(sess, states_batch, angle_action_batch_idx, taptime_action_batch_idx, angle_targets_batch, taptime_targets_batch)

    # taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)
    # return angle_loss, taptime_loss
    return loss

def pretrain_parNN(replay_memory, valid_angles, valid_taptimes, angle_estimator, taptime_estimator, angle_target_estimator, taptime_target_estimator, sess, batch_size=6, discount_factor=0.7, angle_feed = False):
    samples = random.sample(replay_memory, batch_size)
    states_batch, action_batch, reward_batch, next_states_batch, game_state_batch = map(np.array, zip(*samples))
    reward_batch = np.clip(reward_batch/10000, 0, 6)
    # (1,1,4096) (1,2) (1,) (1,)

    done_batch = np.array([1 if (game_state =='LOST' or 'WON') else 0 for game_state in game_state_batch])
    angle_action_batch_idx = np.array([valid_angles.index(action_batch[i][0]) for i in range(batch_size)])
    taptime_action_batch_idx = np.array([valid_taptimes.index(action_batch[i][1]) for i in range(batch_size)])

    # 학습에 넣을 target reward 계산 : angle
    angle_q_values_next = angle_estimator.predict(sess, next_states_batch)
    best_angle_actions = np.argmax(angle_q_values_next, axis=1)
    # 현재의 angle action을 taptime state input으로 넣는다.
    # 다음 state의 best_angle_action을 taptime next state input으로 넣는다.
    if angle_feed:
        current_angle_feed_batch = np.zeros((batch_size, len(valid_angles)), dtype = int)
        next_angle_feed_batch = np.zeros((batch_size, len(valid_angles)), dtype = int)
        for i in range(batch_size):
            current_angle_feed_batch[i][angle_action_batch_idx[i]] = 1
            next_angle_feed_batch[i][best_angle_actions[i]] = 1
        taptime_states_batch = np.concatenate((states_batch, current_angle_feed_batch), axis = 1)
        taptime_next_states_batch = np.concatenate((next_states_batch, next_angle_feed_batch), axis = 1)

    # 학습에 넣을 target reward 계산 : taptime
    if angle_feed:
        taptime_q_values_next = taptime_estimator.predict(sess, taptime_next_states_batch)
    else:
        taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
    # taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
    best_taptime_actions = np.argmax(taptime_q_values_next, axis=1)

    angle_q_values_next_target = angle_target_estimator.predict(sess, next_states_batch)
    if angle_feed:
        taptime_q_values_next_target = taptime_target_estimator.predict(sess, taptime_next_states_batch)
    else:
        taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)
    # taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)

    angle_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        discount_factor * angle_q_values_next_target[np.arange(batch_size), best_angle_actions]

    taptime_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        discount_factor * taptime_q_values_next_target[np.arange(batch_size), best_taptime_actions]

    # Perform gradient descent update
    states_batch = np.array(states_batch)
    angle_loss = angle_estimator.update(sess, states_batch, angle_action_batch_idx, angle_targets_batch)
    if angle_feed:
        taptime_states_batch = np.array(taptime_states_batch)
        taptime_loss = taptime_estimator.update(sess, taptime_states_batch, taptime_action_batch_idx, taptime_targets_batch)
    else:
        taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)
    # taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)
    return angle_loss, taptime_loss

def init_oneshot_onekill(exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    dir_list = []
    png_list = []
    for filename in glob.iglob("%s/*/*/s_1_*_WON.png_seg.png"%(exp_path)):
        next_state = os.path.abspath(filename)
        png_list.append(next_state)
        dir = os.path.dirname(next_state)
        dir_list.append(dir)
        state = glob.glob("%s/s_?.png_seg.png"%dir)
        action, reward= None, None
        with open(os.path.join(dir, 'action'), 'rb') as f:
            action = pickle.load(f)
        with open(os.path.join(dir, 'reward'), 'rb') as f:
            reward = pickle.load(f)
        state = get_feature_4096(model=model_name, img_path=state[0])
        next_state = get_feature_4096(model=model_name, img_path=next_state)
        replay_memory.append([state, action, reward, next_state, 'WON'])
    return replay_memory, dir_list, png_list

def init_twoshot_onekill(exp_path, current_dir, model_name):
    import os, glob, pickle
    replay_memory = []
    dir_list = []
    png_list = []
    for filename in glob.iglob("%s/*/*/s_2_*_WON.png_seg.png"%(exp_path)):

        #shot2에서 이기는 것을 oneshot onekill과 같이 저장
        next_state = os.path.abspath(filename)
        png_list.append(next_state)
        dir = os.path.dirname(next_state)
        dir_list.append(dir)
        state = glob.glob("%s/s_?.png_seg.png"%dir)
        action, reward= None, None
        with open(os.path.join(dir, 'action'), 'rb') as f:
            action = pickle.load(f)
        with open(os.path.join(dir, 'reward'), 'rb') as f:
            reward = pickle.load(f)
        state = get_feature_4096(model=model_name, img_path=state[0])
        next_state = get_feature_4096(model=model_name, img_path=next_state)
        replay_memory.append([state, action, reward, next_state, 'WON'])

        #이 때 이전 shot1의 정보도 저장해야 함.
        dir = os.path.dirname(dir) # 그 폴더의 상위 폴더, 여기에 있는 s0~~ 폴더 안에서 샷도 뽑아야 함
        dir = glob.glob("%s/*_shot0_*"%(dir)) #여기에 있는 s0폴더로 들어왔음!
        dir = dir[0]
        next_state = glob.glob("%s/*_PLAYING.png_seg.png"%dir)
        png_list.append(next_state)
        dir_list.append(dir)
        state = glob.glob("%s/s_?.png_seg.png"%dir)
        action, reward= None, None
        with open(os.path.join(dir, 'action'), 'rb') as f:
            action = pickle.load(f)
        with open(os.path.join(dir, 'reward'), 'rb') as f:
            reward = pickle.load(f)
        state = get_feature_4096(model=model_name, img_path=state[0])
        next_state = get_feature_4096(model=model_name, img_path=next_state[0])
        replay_memory.append([state, action, reward, next_state, 'PLAYING'])

    return replay_memory, dir_list, png_list


def get_valid_angles():
    import csv
    f = open('experience_angles.csv', 'r')
    rdr = csv.reader(f)
    angles = []
    for line in rdr:
        angles.append(int(line[0]))
    f.close()
    angles = list(set(angles))
    return angles

# experience_gathering으로부터 각도셋(experience_angles.csv)으로 pickle 생성
if __name__ == '__main__':
    import inspect, os, csv, pickle
    # from multiprocessing import Pool
    from tensorflow.python.keras.applications.vgg16 import VGG16


    current_path = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(os.path.abspath(current_path))
    EXP_PATH=os.path.join(current_dir,"experiences_parNN_feed_thread")
    vgg16 = VGG16(weights= 'imagenet')

    replay_memory = init_replaymemory_all_levels(EXP_PATH, current_dir, vgg16)

    # with open(os.path.join(EXP_PATH, 'replay_memory_newAll'), 'wb') as f:
    #     pickle.dump(replay_memory, f)
    np.save('replay_memory_with_levels_0615', replay_memory)
    print('Done')

    # angles = [8,10,11,14,17,18,19,20,21,22,23,26,30,31,34,35,36,46,61,65,67,70,75]

    # current_path = inspect.getfile(inspect.currentframe())
    # current_dir = os.path.dirname(os.path.abspath(current_path))
    # EXP_PATH=os.path.join(current_dir,"experiences_gathering")
    # vgg16 = VGG16(weights= 'imagenet')

    # print('Populating replay memory...')
    # # pool = Pool(processes=3)
    # # pool.map(angles[:])
    # # pool.map(init_replaymemory_WithAngleSet, angles)

    # replay_memory = init_replaymemory_WithAngleSet(angles, EXP_PATH, current_dir, vgg16)
    # # replay_memory = (angles, EXP_PATH, current_dir, vgg16)

    # with open(os.path.join(EXP_PATH, 'replay_memoryAll'), 'wb') as f:
    #     pickle.dump(replay_memory, f)
    # print('Done')
