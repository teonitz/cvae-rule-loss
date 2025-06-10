# --- Глобальные параметры архитектуры ---
ROOM_HEIGHT = 16
ROOM_WIDTH = 11
num_tiles = 10 # Количество уникальных тайлов
num_door_features = 4 # Количество условных признаков (верх, низ, лево, право)

LATENT_DIM = 32 # Размерность латентного пространства (можно экспериментировать)
BATCH_SIZE = 64 # Размер батча для обучения
EPOCHS = 200 # Количество эпох обучения (может потребоваться больше)
LEARNING_RATE = 0.001 # Скорость обучения

# --- 1. Энкодер ---
def build_encoder(input_shape_X, input_shape_Y, latent_dim):
    encoder_input_X = keras.Input(shape=input_shape_X, name='room_input')
    encoder_input_Y = keras.Input(shape=input_shape_Y, name='condition_input_encoder')

    y_expanded = layers.Lambda(lambda y: tf.expand_dims(tf.expand_dims(y, 1), 1),
                               name='expand_y', output_shape=(1, 1, input_shape_Y[-1]))(encoder_input_Y)

    y_tiled = layers.Lambda(lambda y_e: tf.tile(y_e, [1, ROOM_HEIGHT, ROOM_WIDTH, 1]),
                            name='tile_y', output_shape=(input_shape_X[0], input_shape_X[1], input_shape_Y[-1]))(y_expanded)

    x_and_y = layers.Concatenate(axis=-1)([encoder_input_X, y_tiled])

    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(x_and_y)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', strides=1, padding='same')(x)

    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    encoder = keras.Model([encoder_input_X, encoder_input_Y], [z_mean, z_log_var], name='encoder')
    return encoder

# --- 2. Декодер ---
def build_decoder(latent_dim, input_shape_Y, output_shape_X):
    decoder_input_z = keras.Input(shape=(latent_dim,), name='z_sample')
    decoder_input_Y = keras.Input(shape=input_shape_Y, name='condition_input_decoder')

    z_and_y = layers.Concatenate(axis=-1)([decoder_input_z, decoder_input_Y])

    start_h = 4
    start_w = 3

    x = layers.Dense(start_h * start_w * 128, activation='relu')(z_and_y)
    x = layers.Reshape((start_h, start_w, 128))(x)

    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)

    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x) # (8,6) -> (16,12)
    x = layers.Cropping2D(cropping=((0, 0), (0, 1)))(x) # Обрезаем один столбец справа, чтобы получить 16x11

    decoder_output = layers.Conv2DTranspose(num_tiles, 3, activation='softmax', padding='same')(x) # (16,11) -> (16,11)

    decoder = keras.Model([decoder_input_z, decoder_input_Y], decoder_output, name='decoder')
    return decoder

# --- 3. Класс CVAE ---
@tf.keras.utils.register_keras_serializable()
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, cvae_loss_layer, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.cvae_loss_layer = cvae_loss_layer

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def call(self, inputs):
        X_batch, Y_batch_cond = inputs

        z_mean, z_log_var = self.encoder([X_batch, Y_batch_cond])

        # Репараметризационный трюк (для генерации/прохода вперед)
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, LATENT_DIM))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        reconstruction = self.decoder([z, Y_batch_cond])
        return reconstruction

    def train_step(self, data):
        (X_batch, Y_batch_cond), X_target_batch = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([X_batch, Y_batch_cond])

            batch_size = tf.shape(z_mean)[0]
            epsilon = tf.random.normal(shape=(batch_size, LATENT_DIM))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            reconstruction = self.decoder([z, Y_batch_cond])

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.categorical_crossentropy(X_target_batch, reconstruction),
                    axis=(1, 2)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        (X_batch, Y_batch_cond), X_target_batch = data

        z_mean, z_log_var = self.encoder([X_batch, Y_batch_cond])

        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, LATENT_DIM))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        reconstruction = self.decoder([z, Y_batch_cond])

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(X_target_batch, reconstruction),
                axis=(1, 2)
            )
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_config(self):
        config = super().get_config()
        encoder_config = tf.keras.utils.serialize_keras_object(self.encoder)
        decoder_config = tf.keras.utils.serialize_keras_object(self.decoder)
        cvae_loss_layer_config = tf.keras.utils.serialize_keras_object(self.cvae_loss_layer)

        config.update({
            "encoder_config": encoder_config,
            "decoder_config": decoder_config,
            "cvae_loss_layer_config": cvae_loss_layer_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop('encoder_config')
        decoder_config = config.pop('decoder_config')
        cvae_loss_layer_config = config.pop('cvae_loss_layer_config')

        encoder = tf.keras.utils.deserialize_keras_object(encoder_config)
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        cvae_loss_layer = tf.keras.utils.deserialize_keras_object(cvae_loss_layer_config)

        return cls(encoder=encoder, decoder=decoder, cvae_loss_layer=cvae_loss_layer, **config)

# Слой выборки по Латенту (Sampling Layer)
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """Использует (z_mean, z_log_var) для сэмплирования z"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# CVAE Потеря (CVAELoss)
@tf.keras.utils.register_keras_serializable()
class CVAELoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CVAELoss, self).__init__(**kwargs)
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, inputs):
        x_true, x_reconstructed, z_mean, z_log_var = inputs

        # 1. Reconstruction loss
        # sparse_categorical_crossentropy ожидает метки (целые числа) для x_true
        # и логиты/вероятности для x_reconstructed.
        # reshape x_true to (batch_size * H * W, 1) and x_reconstructed to (batch_size * H * W, num_classes)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                tf.cast(x_true, dtype=tf.int32), x_reconstructed
            )
        ) * tf.cast(tf.size(x_true), dtype=tf.float32) # Умножаем на количество элементов для получения суммы, как в оригинальной статье VAE

        # 2. KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss) # Усредняем по батчу

        total_loss = reconstruction_loss + kl_loss

        self.add_metric(reconstruction_loss, name="reconstruction_loss")
        self.add_metric(kl_loss, name="kl_loss")
        self.add_metric(total_loss, name="total_loss")
        self.add_loss(total_loss)
        return total_loss

    # Для корректного сохранения и загрузки слоя
    def get_config(self):
        config = super(CVAELoss, self).get_config()
        return config