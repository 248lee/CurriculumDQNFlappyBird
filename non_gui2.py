from deep_q_network import trainNetwork
trainNetwork(1, 3, lock_mode=1, is_simple_actions_locked=True, max_steps=300000, resume_Adam=False, learning_rate=1e-6, event=None, is_colab=True)
