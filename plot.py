import matplotlib.pyplot as plt
import numpy as np

def plot_image_with_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(range(10)[predicted_label],
                                100*np.max(predictions_array),
                                range(10)[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_image(img, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)
    plt.xlabel("{}".format(true_label))

def plot_images(images, labels, num_rows = 3, num_cols = 5):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plot_image(images[i], labels[i])
    plt.tight_layout()
    plt.show()


def plot_predictions(images, labels, predictions, num_rows = 5, num_cols = 3):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    indices = np.random.randint(0, len(images), num_images)
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image_with_prediction(indices[i], predictions[indices[i]], labels, images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(indices[i], predictions[indices[i]], labels)
    plt.tight_layout()
    plt.show()