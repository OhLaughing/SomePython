import matplotlib.pyplot as plt
from keras.datasets import mnist

def showImage(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    showImage(train_images[0])