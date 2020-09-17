from keras import models
from keras import layers
if __name__ == '__main__':
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    