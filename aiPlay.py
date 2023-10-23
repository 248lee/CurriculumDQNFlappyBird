from deep_q_network import play
<<<<<<< HEAD
import numpy as np
nums = []
avgs = []
avgs_mean = 0
avgs_var = 0
i = 0
record = open('record40x40to80x80.txt', 'w')
while True:
    num, avg = play(2, 100000)
    nums.append(num)
    avgs.append(avg)
    np_avgs = np.array(avgs)
    record.write('mean: ' + str(np.mean(np_avgs)))
    record.write('var: ' + str(np.var(np_avgs)))
    record.write('---------------------------------------')
    i += 1
    if i < 20:
        com = input()
        if com == 'exit':
            break
print(nums)
print(avgs)

np_avgs = np.array(avgs)
print('average score')
print('mean: ', np.mean(np_avgs))
print('var: ', np.var(np_avgs))

np_nums = np.array(nums)
print('num of episodes')
print('mean: ', np.mean(np_nums))
print('var: ', np.var(np_nums))



=======
play(1, 10000)
>>>>>>> 7383780ce177d33ca6d2426f0d9fa96de63c6f90
