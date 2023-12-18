from deep_q_network import trainNetwork
trainNetwork(2, 3, False, 200000, resume_Adam=False, learning_rate=1e-5, event=None, is_colab=True)
