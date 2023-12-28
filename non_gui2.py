from deep_q_network import trainNetwork
trainNetwork(2, 3, lock_mode=2, is_simple_actions_locked=False, is_activate_boss_memory=True, max_steps=350000, resume_Adam=False, learning_rate=1e-5, event=None, is_colab=True)
