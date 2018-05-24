import numpy as np

import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
import numpy as np

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
        # pdb.set_trace()
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon) # ? 
        return A
    return policy_fn

def preprocessing(state_img, fe_model):
    pass

def get_feature_4096(model, img_path):
    '''
    vgg16 = VGG16(weights= 'imagenet', include_top= False) # output: 7*7*512
    vgg16 = VGG16(weights= 'imagenet', include_top= True) # output: 1000*1
    '''
    model_4096 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

    img = image.load_img(img_path, target_size=(224, 224)) # 우선 image.load_img로 강제 resize
    # pdb.set_trace()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model_4096.predict(x)

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

def get_score_after_shot(parser, comm_socket, last_score):
    end_image = None
    no_change_count=0
    start_time = time.time()
    
    save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
    screenshot = comm_do_screenshot(comm_socket, save_path=save_path)
    start_score = parser.get_score_in_game(save_path)

    pigs=[]
    birds=[]
    # Shooting is done. Check the score
    while True:
        
        # if the game has ended, end checking
        state = comm_get_state(comm_socket)
        if state == WON or state == LOST:
            break

        # check the status of the screenshot
        save_path = "%s/screenshots/screenshot_%d.png" % (current_dir, int(time.time()*1000))
        screenshot = comm_do_screenshot(comm_socket, save_path=save_path)
        score   = parser.get_score_in_game(save_path)
        pigs    = parser.get_pigs(save_path)
        birds   = parser.get_birds(save_path)

        # if the new score is less than the last score, something went wrong, just return the last status
        if last_score > score:
            break
        # if the new score is equal to the last score. count the time
        elif last_score==score:
            end_image = screenshot
            time.sleep(1000)
            no_change_count+=1

        # if the new score is higher than the last score, keep tracking
        else:
            end_image = screenshot
            last_score = score
            no_change_count = 0

        # if the no change count and passed time are enough, return.
        if no_change_count>=10:

            # if the game has ended (pig=0), and there are remaining birds, wait for the birds bonus.
            if len(pigs)==0 and len(birds)>=2:
                no_change_count-= 40
            else:
                break
    is_terminal = False
    if len(pigs)==0 or len(birds)==0:
        is_terminal = True
    return last_score-start_score, last_score, save_path, is_terminal