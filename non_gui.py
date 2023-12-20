from deep_q_network import trainNetwork
trainNetwork(1, 3, lock_mode=2, is_simple_actions_locked=True, max_steps=200000, resume_Adam=False, learning_rate=5e-6, event=None, is_colab=True)
