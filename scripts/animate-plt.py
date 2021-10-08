import matplotlib.pyplot as plt
import numpy as np

plt.clf()
plt.close()
for i in range(10):
    fig, axs = plt.subplots(2, len(ncm.V), sharex=True, sharey=True, figsize=(12, 5))
    for a in axs.flatten():
        a.bar([0,1],np.random.rand(2))
        a.set_xlim(0,1)
        a.set_ylim(0,1)
    #plt.draw()
    plt.pause(0.001)
    #input("Press [enter] to continue.")
plt.show()
for i in range(100):
    plt.close()