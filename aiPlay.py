from deep_q_network import play
import numpy as np
nums = []
avgs = []
avgs_mean = 0
avgs_var = 0
i = 0
while True:
    num, avg = play(2, 100000)
    nums.append(num)
    avgs.append(avg)
    np_avgs = np.array(avgs)
    record = open('record30x30to60x60.txt', 'a')
    record.write(str(avgs))
    record.write('\nmean: ' + str(np.mean(np_avgs)))
    record.write('   var: ' + str(np.var(np_avgs)))
    record.write('\n---------------------------------------\n')
    record.close()
    i += 1
    if i >= 30:
        com = input()
        if com == 'exit':
            break
        else:
            i = 0
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

