import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
i=1
plt.matshow(digits.images[i])
print(digits.target[i])
plt.show()