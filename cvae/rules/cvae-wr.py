# --- Глобальные параметры архитектуры ---
ROOM_HEIGHT = 16 # Высота комнаты
ROOM_WIDTH = 11  # Ширина комнаты
num_tiles = 10   # Количество уникальных тайлов (символов)
num_door_features = 4 # Количество условных признаков для дверей (верх, низ, лево, право)

LATENT_DIM = 32 # Размерность латентного пространства
BATCH_SIZE = 64 # Размер батча для обучения
EPOCHS = 500 # Количество эпох обучения
LEARNING_RATE = 0.001 # Скорость обучения
RULE_LOSS_WEIGHT = 10 # Вес для члена потерь, связанного с правилами. Этот параметр нужно настраивать.

token_map = {
    'F': 0, 'B': 1, 'M': 2, 'P': 3, 'O': 4,
    'I': 5, 'D': 6, 'S': 7, 'W': 8, '-': 9
}
int_to_token = {v: k for k, v in token_map.items()}

VOID_ID = tf.constant(token_map['-'], dtype=tf.int32)
WALL_ID = tf.constant(token_map['W'], dtype=tf.int32)
DOOR_ID = tf.constant(token_map['D'], dtype=tf.int32)
FLOOR_ID = tf.constant(token_map['F'], dtype=tf.int32)

PASSABLE_TILES_IDS = tf.constant([token_map['F'], token_map['M'], token_map['S'], token_map['O']], dtype=tf.int32)
OBJECT_TILES_IDS = tf.constant([token_map['M'], token_map['P'], token_map['O'], token_map['I'], token_map['S']], dtype=tf.int32)

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
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)

    x = layers.Cropping2D(cropping=((0, 0), (0, 1)))(x)

    decoder_output = layers.Conv2DTranspose(num_tiles, 3, activation='softmax', padding='same')(x)

    decoder = keras.Model([decoder_input_z, decoder_input_Y], decoder_output, name='decoder')
    return decoder

# --- Функция для вычисления потерь на основе логических правил ---
def calculate_rule_loss(generated_outputs_logits, target_door_info_flat, room_height, room_width, token_map,
                        enable_void_rule, enable_wall_integrity_rule,
                        enable_object_placement_rule, enable_door_environment_rule):
    generated_token_ids = tf.argmax(generated_outputs_logits, axis=-1, output_type=tf.int32)

    batch_size = tf.shape(generated_token_ids)[0]
    rule_penalties = tf.zeros(batch_size, dtype=tf.float32)

    if enable_void_rule:
        penalty_void = tf.cast(tf.reduce_sum(tf.cast(generated_token_ids == VOID_ID, tf.float32), axis=[1, 2]), tf.float32)
        rule_penalties += penalty_void

    if enable_wall_integrity_rule:

        top_row = generated_token_ids[:, 0, :]
        penalty_top = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(top_row != WALL_ID, top_row != DOOR_ID), tf.float32), axis=1), tf.float32)
        rule_penalties += penalty_top

        bottom_row = generated_token_ids[:, room_height - 1, :]
        penalty_bottom = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(bottom_row != WALL_ID, bottom_row != DOOR_ID), tf.float32), axis=1), tf.float32)
        rule_penalties += penalty_bottom

        left_col = generated_token_ids[:, 1:room_height-1, 0]
        penalty_left = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(left_col != WALL_ID, left_col != DOOR_ID), tf.float32), axis=1), tf.float32)
        rule_penalties += penalty_left

        right_col = generated_token_ids[:, 1:room_height-1, room_width - 1]
        penalty_right = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(right_col != WALL_ID, right_col != DOOR_ID), tf.float32), axis=1), tf.float32)
        rule_penalties += penalty_right

    if enable_object_placement_rule:

        expanded_generated_token_ids = tf.expand_dims(generated_token_ids, axis=-1)
        matches_any_object_id = tf.equal(expanded_generated_token_ids, OBJECT_TILES_IDS)
        is_object_tile = tf.reduce_any(matches_any_object_id, axis=-1)

        is_invalid_placement_tile = tf.logical_or(generated_token_ids == WALL_ID, generated_token_ids == VOID_ID)

        penalty_object_placement = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(is_object_tile, is_invalid_placement_tile), tf.float32), axis=[1, 2]), tf.float32)
        rule_penalties += penalty_object_placement

    if enable_door_environment_rule:

        horizontal_door_cols = tf.constant([room_width // 2 - 1, room_width // 2, room_width // 2 + 1], dtype=tf.int32)
        vertical_door_rows = tf.constant([room_height // 2 - 1, room_height // 2], dtype=tf.int32)

        top_door_expected = tf.cast(target_door_info_flat[:, 0], tf.bool)
        actual_top_door_tiles = tf.gather(generated_token_ids[:, 1, :], horizontal_door_cols, axis=1)
        is_actual_top_door_present = tf.reduce_all(tf.equal(actual_top_door_tiles, DOOR_ID), axis=1)
        penalty_missing_top_door = tf.cast(tf.logical_and(top_door_expected, tf.logical_not(is_actual_top_door_present)), tf.float32)
        rule_penalties += penalty_missing_top_door * 5.0

        floor_below_top_door_tiles = tf.gather(generated_token_ids[:, 2, :], horizontal_door_cols, axis=1)
        is_floor_below_top_door = tf.reduce_all(tf.equal(floor_below_top_door_tiles, FLOOR_ID), axis=1)
        penalty_bad_top_door_env = tf.cast(tf.logical_and(is_actual_top_door_present, tf.logical_not(is_floor_below_top_door)), tf.float32)
        rule_penalties += penalty_bad_top_door_env * 2.0

        bottom_door_expected = tf.cast(target_door_info_flat[:, 1], tf.bool)
        actual_bottom_door_tiles = tf.gather(generated_token_ids[:, room_height - 2, :], horizontal_door_cols, axis=1)
        is_actual_bottom_door_present = tf.reduce_all(tf.equal(actual_bottom_door_tiles, DOOR_ID), axis=1)

        penalty_missing_bottom_door = tf.cast(tf.logical_and(bottom_door_expected, tf.logical_not(is_actual_bottom_door_present)), tf.float32)
        rule_penalties += penalty_missing_bottom_door * 5.0

        floor_above_bottom_door_tiles = tf.gather(generated_token_ids[:, room_height - 3, :], horizontal_door_cols, axis=1)
        is_floor_above_bottom_door = tf.reduce_all(tf.equal(floor_above_bottom_door_tiles, FLOOR_ID), axis=1)
        penalty_bad_bottom_door_env = tf.cast(tf.logical_and(is_actual_bottom_door_present, tf.logical_not(is_floor_above_bottom_door)), tf.float32)
        rule_penalties += penalty_bad_bottom_door_env * 2.0

        left_door_expected = tf.cast(target_door_info_flat[:, 2], tf.bool)
        actual_left_door_tiles = tf.gather(generated_token_ids[:, :, 1], vertical_door_rows, axis=1)
        is_actual_left_door_present = tf.reduce_all(tf.equal(actual_left_door_tiles, DOOR_ID), axis=1)

        penalty_missing_left_door = tf.cast(tf.logical_and(left_door_expected, tf.logical_not(is_actual_left_door_present)), tf.float32)
        rule_penalties += penalty_missing_left_door * 5.0

        floor_right_of_left_door_tiles = tf.gather(generated_token_ids[:, :, 2], vertical_door_rows, axis=1)
        is_floor_right_of_left_door = tf.reduce_all(tf.equal(floor_right_of_left_door_tiles, FLOOR_ID), axis=1)
        penalty_bad_left_door_env = tf.cast(tf.logical_and(is_actual_left_door_present, tf.logical_not(is_floor_right_of_left_door)), tf.float32)
        rule_penalties += penalty_bad_left_door_env * 2.0

        right_door_expected = tf.cast(target_door_info_flat[:, 3], tf.bool)
        actual_right_door_tiles = tf.gather(generated_token_ids[:, :, room_width - 2], vertical_door_rows, axis=1)
        is_actual_right_door_present = tf.reduce_all(tf.equal(actual_right_door_tiles, DOOR_ID), axis=1)

        penalty_missing_right_door = tf.cast(tf.logical_and(right_door_expected, tf.logical_not(is_actual_right_door_present)), tf.float32)
        rule_penalties += penalty_missing_right_door * 5.0

        floor_left_of_right_door_tiles = tf.gather(generated_token_ids[:, :, room_width - 3], vertical_door_rows, axis=1)
        is_floor_left_of_right_door = tf.reduce_all(tf.equal(floor_left_of_right_door_tiles, FLOOR_ID), axis=1)
        penalty_bad_right_door_env = tf.cast(tf.logical_and(is_actual_right_door_present, tf.logical_not(is_floor_left_of_right_door)), tf.float32)
        rule_penalties += penalty_bad_right_door_env * 2.0

    return tf.reduce_mean(rule_penalties)

# --- 3. Класс CVAE ---
@tf.keras.utils.register_keras_serializable()
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.rule_loss_tracker = keras.metrics.Mean(name='rule_loss')

        self._enable_void_rule = True
        self._enable_wall_integrity_rule = True
        self._enable_object_placement_rule = True
        self._enable_door_environment_rule = True

    def set_rule_flags(self, void=True, wall_integrity=True, object_placement=True, door_environment=True):
        self._enable_void_rule = void
        self._enable_wall_integrity_rule = wall_integrity
        self._enable_object_placement_rule = object_placement
        self._enable_door_environment_rule = door_environment


    def call(self, inputs):
        X_batch, Y_batch_cond = inputs
        z_mean, z_log_var = self.encoder([X_batch, Y_batch_cond])

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

            rule_loss = calculate_rule_loss(
                reconstruction, Y_batch_cond, ROOM_HEIGHT, ROOM_WIDTH, token_map,
                enable_void_rule=self._enable_void_rule,
                enable_wall_integrity_rule=self._enable_wall_integrity_rule,
                enable_object_placement_rule=self._enable_object_placement_rule,
                enable_door_environment_rule=self._enable_door_environment_rule
            )

            total_loss = reconstruction_loss + kl_loss + RULE_LOSS_WEIGHT * rule_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.rule_loss_tracker.update_state(rule_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'rule_loss': self.rule_loss_tracker.result(),
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

        rule_loss = calculate_rule_loss(
            reconstruction, Y_batch_cond, ROOM_HEIGHT, ROOM_WIDTH, token_map,
            enable_void_rule=self._enable_void_rule,
            enable_wall_integrity_rule=self._enable_wall_integrity_rule,
            enable_object_placement_rule=self._enable_object_placement_rule,
            enable_door_environment_rule=self._enable_door_environment_rule
        )

        total_loss = reconstruction_loss + kl_loss + RULE_LOSS_WEIGHT * rule_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.rule_loss_tracker.update_state(rule_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'rule_loss': self.rule_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.rule_loss_tracker,
        ]

    def get_config(self):
        config = super().get_config()
        encoder_config = tf.keras.utils.serialize_keras_object(self.encoder)
        decoder_config = tf.keras.utils.serialize_keras_object(self.decoder)

        config.update({
            "encoder_config": encoder_config,
            "decoder_config": decoder_config,
            "enable_void_rule": self._enable_void_rule,
            "enable_wall_integrity_rule": self._enable_wall_integrity_rule,
            "enable_object_placement_rule": self._enable_object_placement_rule,
            "enable_door_environment_rule": self._enable_door_environment_rule,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop('encoder_config')
        decoder_config = config.pop('decoder_config')

        encoder = tf.keras.utils.deserialize_keras_object(encoder_config, safe_mode=False)
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config, safe_mode=False)

        void_rule = config.pop('enable_void_rule', True)
        wall_integrity_rule = config.pop('enable_wall_integrity_rule', True)
        object_placement_rule = config.pop('enable_object_placement_rule', True)
        door_environment_rule = config.pop('enable_door_environment_rule', True)

        instance = cls(encoder=encoder, decoder=decoder, **config)

        instance.set_rule_flags(
            void=void_rule,
            wall_integrity=wall_integrity_rule,
            object_placement=object_placement_rule,
            door_environment=door_environment_rule
        )
        return instance