import numpy as np
import scipy.stats

a = np.array([0,2,2,4,4,6,6,8,8,10,10])
b = np.array([])
t = 0
for t in range(len(a)) :
    t = t + 1
    if t < 7 :
        moded_a = scipy.stats.mode(a[t:t+2])[0]
        b = np.concatenate((b, moded_a), axis = 0)

print(b)