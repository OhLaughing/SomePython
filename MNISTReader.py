import os
import struct

import matplotlib.pyplot as plt
import numpy as np

'''
将MNIST的数据通过matplot显示出来
'''


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# images 为60000*784
# labels 为60000*1
images, labels = load_mnist(r'E:\workspace\pdf\李航\mnist')
print(type(images))
print(type(labels))
print(images.ndim)
print(images.size)
print(images.shape)
print(labels.shape)
a = images[0]
print(len(a))
b = a.reshape([28, 28])
print(b.shape)
print(b)
print(labels[0])

fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    tmp = images[i]
    for j in range(len(tmp)):
        if (tmp[j] > 0):
            tmp[j] = 255
    img = tmp.reshape(28, 28)
    print(tmp)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
# ax[0].set_yticks([])
plt.tight_layout()
plt.show()
