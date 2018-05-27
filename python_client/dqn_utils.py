import time
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
import numpy as np
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
        A = np.ones(nA, dtype=float) * epsilon / nA # action 수 만큼의 길이
        # pdb.set_trace()
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        # q_values = estimator.predict(sess, observation)[0]
        # pdb.set_trace()
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
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

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    # pdb.set_trace()
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def get_slingshot_refpoint(slingshot):
	X_OFFSET = 0.5
	Y_OFFSET = 0.65
	x = slingshot[0]+slingshot[2]*X_OFFSET
	y = slingshot[1]+slingshot[2]*Y_OFFSET
	return (x,y)

def get_score_after_shot(current_dir, parser, comm_socket, start_score):
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
            if sleepcount>=10:
                save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
                end_image = comm.comm_do_screenshot(comm_socket, save_path=save_path)
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
