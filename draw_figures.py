import numpy as np
import matplotlib.pyplot as plt

G_loss = np.loadtxt('G_loss.txt', delimiter=' ')
D_loss = np.loadtxt('D_loss.txt', delimiter=' ')

plt.plot(D_loss)
plt.show()