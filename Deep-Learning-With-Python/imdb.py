from keras import models
from keras import layers
from keras import optimizers
if __name__ == '__main__':
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    