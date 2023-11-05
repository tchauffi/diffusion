from io import BytesIO

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class CustomTensorboard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, dataset):
        super().__init__()
        self.log_dir = log_dir
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        super().on_epoch_end(epoch, logs, **kwargs)

        images = next(iter(self.dataset))

        _, _, reconstructions = self.model(images)
        reconstructions = reconstructions.numpy()

        fig = plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(np.hstack((images[i], reconstructions[i])))
            plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.image("Reconstruction", image, step=epoch)