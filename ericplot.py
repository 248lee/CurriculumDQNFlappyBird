
import json
import matplotlib.pyplot as plt
average_rewards = []
with open("avgrewards.txt",'r') as f:
  for nums in f.readlines():
    for num in json.loads(nums):
      average_rewards.append(num)
plt.xlabel('Trainingsteps(per thousand)')
plt.ylabel('AverageRewards')
plt.plot(average_rewards)
plt.savefig('reward_plot')