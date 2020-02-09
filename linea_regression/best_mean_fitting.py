import numpy as np
import matplotlib.pyplot as plt

xd = np.asarray([1, 2, 3, 4, 5, 6])
yd = np.asarray([5, 4, 6, 5, 6, 7])


def best_fit(x, y):
    m = ((np.mean(x) * np.mean(y)) - (np.mean(x * y))) / (np.mean(x) ** 2) - (np.mean(x * x))
    b = (np.mean(y)) - (m * np.mean(x))
    return m, b


m, b = best_fit(xd, yd)

yh = []
for x in xd:
    yh.append(m * x + b)

print(m)
print(b)

plt.scatter(xd, yd)
plt.plot(xd, yd)
plt.show()
