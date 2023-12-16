import os
import matplotlib.pyplot as plt
for i in range(3):
    ctr = 0
    lines = []
    lines_sparse = []
    if os.path.exists('./Qvalues/Q' + str(i) + '.txt'):
        file = open('./Qvalues/Q' + str(i) + '.txt', 'r')
        if os.path.getsize('./Qvalues/Q' + str(i) + '.txt'):
        # Read all lines from the file and convert them to floats
            for line in file:
              lines.append(float(line.strip()))
              if ctr % 1000 == 0:
                lines_sparse.append(float(line.strip()))
              ctr += 1

            plt.plot(range(len(lines)), lines, label='Q' + str(i))
plt.legend()
plt.savefig("Q0_plot.png")