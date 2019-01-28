'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def create_mnist_cnn_model(num_classes, input_shape, lr=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    if lr is None:
        optimizer=keras.optimizers.Adadelta()
    else:
        optimizer=keras.optimizers.Adam(lr=lr),

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def prepare_data():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape


def sparse_mnist_generator_nonzero(x_train, y_train, batch_size, sparsity, shuffle=True):
    batch_index = np.arange(batch_size)
    while True:
        # generate the batch data full of zeros
        batch = np.zeros((batch_size, *x_train.shape[1:]), dtype='float32')

        # sample random images from the dataset to get a batch
        if shuffle:
            rand_batch = np.random.randint(0, x_train.shape[0], batch_size)
        else:
            rand_batch = np.arange(batch_size)

        # sample random indices of the images
        for b, img in enumerate(rand_batch):
            for i in range(sparsity):
                pixel_val = 0

                # keep sampling random pairs of indices until a non-zero pixel found
                while pixel_val == 0:
                    x, y = np.random.randint(0, x_train.shape[1], 2)
                    pixel_val = x_train[img, x, y][0]

                batch[b, x, y, 0] = pixel_val

        yield batch, y_train[rand_batch]


def eval_generator(val_data_generator, judge, num_repetitions):
    data_x_sparse, data_y = next(val_data_generator)
    true_categories = data_y.argmax(axis=1)

    # calculate predictions by the judge
    accuracies = []
    for i in range(num_repetitions):
        # for validation we are not shuffling the samples -> we can reuse the true_categories and predictions
        data_x_sparse, _ = next(val_data_generator)

        predictions = judge.predict(data_x_sparse).argmax(axis=1)

        acc = (predictions == true_categories).sum() / predictions.shape[0]
        accuracies.append(acc)

    return accuracies


def eval_judge(predictions, true_categories, adversary_precommit):
    x_indices = np.arange(predictions.shape[0])
    accuracy = (predictions[x_indices, true_categories] > predictions[x_indices, adversary_precommit]).sum() / predictions.shape[0]
    return accuracy


def eval_precommit(data_x, data_y, model):
    true_categories = data_y.argmax(axis=1)

    precommit = np.random.randint(0, num_classes, data_y.shape[0])

    equal_selections = precommit == true_categories
    precommit[equal_selections] = precommit[equal_selections] + 1

    # fix overflow
    precommit[precommit == num_classes] = 0

    predictions = model.predict(data_x)

    accuracy = eval_judge(predictions, true_categories, precommit)
    return accuracy


def eval_precommit_generator(val_data_generator, model, num_repetitions):
    accuracies = []
    for i in range(num_repetitions):
        data_x, data_y = next(val_data_generator)
        acc = eval_precommit(data_x, data_y, model)
        accuracies.append(acc)

    return accuracies


def calc_adversary(data_x, model):
    predictions = model.predict(data_x)
    adversary = predictions.argsort(axis=1)
    # -1th column is the best guess, -2nd is the 2nd best guess
    return adversary[:,-1], adversary[:,-2]


def eval_precommit_adversarial_generator(data_x, val_data_generator, judge, adversarial_model, num_repetitions):
    data_x_sparse, data_y = next(val_data_generator)
    true_categories = data_y.argmax(axis=1)

    adversary_best, adversary_precommit = calc_adversary(data_x, adversarial_model)

    # check if the 2nd best category category according to the adversary is actually by chance the true category
    equal_selections = adversary_precommit == true_categories

    # switch these to the actual best guess of the adversary
    adversary_precommit[equal_selections] = adversary_best[equal_selections]

    # calculate predictions by the judge
    accuracies = []
    for i in range(num_repetitions):
        # for validation we are not shuffling the samples -> we can reuse the true_categories and predictions
        data_x_sparse, _ = next(val_data_generator)

        predictions = judge.predict(data_x_sparse)

        acc = eval_judge(predictions, true_categories, adversary_precommit)
        accuracies.append(acc)

    return accuracies