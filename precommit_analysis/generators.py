import numpy as np


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


def eval_precommit(data_x, data_y, model, num_classes):
    true_categories = data_y.argmax(axis=1)

    precommit = np.random.randint(0, num_classes, data_y.shape[0])

    equal_selections = precommit == true_categories
    precommit[equal_selections] = precommit[equal_selections] + 1

    # fix overflow
    precommit[precommit == num_classes] = 0

    predictions = model.predict(data_x)

    accuracy = eval_judge(predictions, true_categories, precommit)
    return accuracy


def eval_precommit_generator(val_data_generator, model, num_classes, num_repetitions):
    accuracies = []
    for i in range(num_repetitions):
        data_x, data_y = next(val_data_generator)
        acc = eval_precommit(data_x, data_y, model, num_classes)
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


def eval_optimal_adversary_generator(val_data_generator, judge, num_repetitions):
    # calculate true categories
    data_x_sparse, data_y = next(val_data_generator)
    true_categories = data_y.argmax(axis=1)
    accuracies = []
    # we have a noisy judge, so we need repetitions to find variance
    for i in range(num_repetitions):
        data_x_sparse, _ = next(val_data_generator)

        predictions = judge.predict(data_x_sparse)

        adversary = predictions.argsort(axis=1)
        adversary_precommit = adversary[:, -1]

        equal_selections = adversary_precommit == true_categories
        adversary_precommit[equal_selections] = adversary[equal_selections, -2]

        acc = eval_judge(predictions, true_categories, adversary_precommit)
        accuracies.append(acc)

    return accuracies