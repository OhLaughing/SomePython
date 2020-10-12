from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

if __name__ == '__main__':
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))


    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    model.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, train_labels, epochs=4, batch_size=512)
    resut = model.predict(x_test, test_labels)
    print(resut)