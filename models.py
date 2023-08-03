import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, ReLU
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Reshape
from keras.layers import LSTM, Embedding, LeakyReLU, Conv2DTranspose
from keras.layers import InputLayer, GRU
from keras.utils import to_categorical
import matplotlib.pyplot as plt

DROPOUT = 0.6
RC_DROPOUT = 0.1

class CNN(tf.keras.Model):
    """
    implement Convolutional Neural Networks using TensorFlow
    """
    def __init__(self):
        super().__init__()

        self.cnn_layers = Sequential([

            # input size: 6960 x 250 x 1 x 22

            # convolution layer 1
            Conv2D(25, kernel_size=(10, 1), strides=1, padding='same', activation='elu', input_shape=(250, 1, 22)),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 2
            Conv2D(50, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 3
            Conv2D(100, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 4
            Conv2D(200, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT)
            
        ], name='CNN_layers')
        
        self.fc = Sequential([
            Flatten(),
            # ReLU(),
            Dense(4, activation='softmax')
        ], name='FC_layers')

    def call(self, x, **kwargs):

        x = self.cnn_layers(x)
        out = self.fc(x)

        return out
    
class RNN_LSTM(tf.keras.Model):
    """
    implement Recurrent Neural Networks using TensorFlow LSTM
    """
    def __init__(self):
        super().__init__()

        self.rnn_layers = Sequential([

            Flatten(),
            Dense(100),
            Reshape((100, 1)),
            LSTM(128, dropout=DROPOUT, recurrent_dropout=0.1, input_shape=(100, 1)),        
            Dense(4, activation='softmax')
        ], name='LSTM_layers')
        
    def call(self, x, **kwargs):

        out = self.rnn_layers(x)

        return out

class RNN_GRU(tf.keras.Model):
    """
    implement Recurrent Neural Networks using TensorFlow GRU
    """
    def __init__(self):
        super().__init__()

        self.rnn_layers = Sequential([

            Flatten(),
            Dense(100),
            Reshape((100, 1)),
            GRU(128, dropout=DROPOUT, recurrent_dropout=0.1, input_shape=(100, 1)),
            Dense(4, activation='softmax')
        ], name='GRU_layers')
        
    def call(self, x, **kwargs):

        out = self.rnn_layers(x)

        return out

class CNN_LSTM(tf.keras.Model):
    """
    implement hybrid model [CNN - LSTM] using TensorFlow
    """
    def __init__(self):
        super().__init__()

        self.crnn_layers = Sequential([

            # input size: 6960 x 250 x 1 x 22

            # convolution layer 1
            Conv2D(25, kernel_size=(10, 1), strides=1, padding='same', activation='elu', input_shape=(250, 1, 22)),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 2
            Conv2D(50, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 3
            Conv2D(100, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 4
            Conv2D(200, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # RNN (LSTM)
            Flatten(),
            Dense(100),
            Reshape((100, 1)),
            LSTM(128, dropout=DROPOUT, recurrent_dropout=RC_DROPOUT, input_shape=(100, 1)),        
            Dense(4, activation='softmax')

        ], name='CNN-LSTM_layers')
        
    def call(self, x, **kwargs):

        out = self.crnn_layers(x)

        return out

class CNN_GRU(tf.keras.Model):
    """
    implement hybrid model [CNN - GRU] using TensorFlow
    """
    def __init__(self):
        super().__init__()

        self.cgnn_layers = Sequential([

            # input size: 6960 x 250 x 1 x 22

            # convolution layer 1
            Conv2D(25, kernel_size=(10, 1), strides=1, padding='same', activation='elu', input_shape=(250, 1, 22)),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 2
            Conv2D(50, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 3
            Conv2D(100, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # convolution layer 4
            Conv2D(200, kernel_size=(10, 1), strides=1, padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            BatchNormalization(),
            Dropout(rate=DROPOUT),

            # RNN (GRU)
            Flatten(),
            Dense(100),
            Reshape((100, 1)),
            GRU(128, dropout=DROPOUT, recurrent_dropout=RC_DROPOUT, input_shape=(100, 1)),        
            Dense(4, activation='softmax')

        ],name='CNN-GRU_layers')
        
    def call(self, x, **kwargs):

        out = self.cgnn_layers(x)
        return out

class CVAE(tf.keras.Model):
    """
    implement Convolutional Variational Autoencoder using TensorFlow
    """
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.encoder = Sequential([


            InputLayer(input_shape=(22, 250, 1)),
            Conv2D(
                filters=32, kernel_size=(1, 20), strides=(1, 10), activation='relu'),
            Conv2D(
                filters=64, kernel_size=(22, 1), activation='relu'),
            Flatten(),
            # No activation
            Dense(latent_dim + latent_dim),


        ],name='encoder')

        self.decoder = Sequential([

            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=1 * 49 * 64, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(1, 49, 64)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(22, 1), padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=(1, 20), strides=(1, 10), padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),

        ], name='decoder')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
