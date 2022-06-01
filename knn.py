import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils import compute_accuracy, get_train_test_dataset

(train_images, train_labels), (test_images,
                               test_labels) = get_train_test_dataset(size=128)

train_image_data = train_images.reshape(train_images.shape[0], -1)
assert train_image_data.shape == (100, 128 * 128 * 3)

test_image_data = test_images.reshape(test_images.shape[0], -1)
assert test_image_data.shape == (20, 128 * 128 * 3)


def knn_plot():
    ks = list(range(1, 21))
    train_accs = []
    test_accs = []

    for k in ks:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            n_jobs=-1,
        ).fit(
            train_image_data,
            train_labels,
        )
        train_result = knn.predict(train_image_data)
        test_result = knn.predict(test_image_data)

        train_accuracy = compute_accuracy(train_labels, train_result)
        test_accuracy = compute_accuracy(test_labels, test_result)

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)

    from matplotlib.ticker import MaxNLocator
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('K Nearest Neighbor')
    plt.plot(ks, train_accs, label='train')
    plt.plot(ks, test_accs, label='test')
    plt.legend()
    plt.savefig('knn.png')


knn_plot()
