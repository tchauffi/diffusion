import tensorflow as tf

from tensorflow.keras import layers, models, losses, optimizers, metrics

from tensorflow.keras import backend as K

def Sampling():
    def apply(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var / 2) * epsilon

    return apply


def Encoder(input_shape, latent_dim, layer_widths=[128, 128, 128, 128]):
    input = layers.Input(shape=input_shape)
    x = input
    for width in layer_widths:
        x = layers.Conv2D(width, kernel_size=3, padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    shape_before_flattening = K.int_shape(x)[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    return models.Model(input, [z_mean, z_log_var, z])


def Decoder(input_shape, latent_dim, layer_widths=[128, 128, 128, 128]):
    input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(1024)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense((K.prod((4, 4 , 128))), activation='relu')(x)
    x = layers.Reshape((4, 4 , 128))(x)
    for width in layer_widths:
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)

    return models.Model(input, x)


class VAE(models.Model):
    def __init__(self, input_shape, latent_dim, beta= 2000):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)
        self.beta = beta    

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                self.beta * losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data): 
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            self.beta * losses.mean_squared_error(data, reconstruction)
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

        

    def compile(self, optimizer, **kwargs):
        super(VAE, self).compile(optimizer, **kwargs)
        self.optimizer = optimizer

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
        self.encoder.save(filepath + '_encoder', overwrite, include_optimizer, save_format, signatures, options)
        self.decoder.save(filepath + '_decoder', overwrite, include_optimizer, save_format, signatures, options)

    def load(self, filepath, compile=True, options=None):
        self.encoder = models.load_model(filepath + '_encoder', compile, options)
        self.decoder = models.load_model(filepath + '_decoder', compile, options)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary(line_length, positions, print_fn)
        self.decoder.summary(line_length, positions, print_fn)



    