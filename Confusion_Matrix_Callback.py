__author__ = 'Bastien Rigaud'

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import io
import itertools
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt


class Add_Confusion_Matrix(Callback):
    def __init__(self, log_dir, validation_steps, validation_data=None, class_names=[], frequency=5):
        super(Add_Confusion_Matrix, self).__init__()
        if validation_data is None:
            AssertionError('Need to provide validation data')
        self.validation_data = iter(validation_data)
        self.validation_steps = validation_steps
        self.class_names = class_names
        self.frequency=frequency
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'val_confusion_matrix'))

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0)
        # Closing the figure prevents it from being displayed
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def plot_confusion_matrix(self, cm, class_names, normalize=True):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # # Compute the labels from the normalized confusion matrix.
            # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap='magma')

        if normalize:
            sm_cm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=1.0))
            sm_cm.set_array([])
            plt.colorbar(sm_cm, ticks=np.arange(0, 1.0+0.2, 0.2))

        plt.title("Confusion matrix")
        plt.colorbar(shrink=0.5)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        if not normalize:
            labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        else:
            labels = np.around(cm, decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "black" if cm[i, j] > threshold else "white"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def log_confusion_matrix(self):
        # Use the model to predict the values from the validation dataset.
        val_labels = []
        val_preds = []
        print('Running confusion matrix for the entire validation dataset...')
        for batch_index in range(self.validation_steps):
            val_images, val_labels_raw = next(self.validation_data)
            val_labels += list(np.argmax(val_labels_raw, axis=-1)[0])
            val_pred_raw = self.model.predict(val_images)
            val_preds += list(np.argmax(val_pred_raw, axis=-1)[0])

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(val_labels, val_preds)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = self.plot_to_image(figure)
        return cm_image

    def on_epoch_end(self, epoch, logs=None):
        # Log the confusion matrix as an image summary.
        if self.frequency != 0 and epoch != 0 and epoch % self.frequency == 0:
            with self.file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", self.log_confusion_matrix(), step=epoch)
