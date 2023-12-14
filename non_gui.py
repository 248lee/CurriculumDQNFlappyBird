from deep_q_network import trainNetwork
trainNetwork(1, 2, False, 2010, resume_Adam=True, learning_rate=1e-6, event=None, is_colab=True)
trainNetwork(1, False, 1010, resume_Adam=True, learning_rate=1e-6, event=None, is_colab=True)
