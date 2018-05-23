import numpy as np

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