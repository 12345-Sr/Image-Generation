import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# CONFIG
# =============================
BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 20
NOISE_DIM = 100
NUM_EXAMPLES = 16

os.makedirs("generated_images", exist_ok=True)

# =============================
# DATA
# =============================
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_train = (x_train - 127.5) / 127.5

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# =============================
# MODELS
# =============================
def build_generator():
    return tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7,7,256)),
        layers.Conv2DTranspose(128, 5, 1, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 5, 2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, 5, 2, padding="same",
                               use_bias=False, activation="tanh")
    ])

def build_discriminator():
    return tf.keras.Sequential([
        layers.Conv2D(64, 5, 2, padding="same", input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, 2, padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])

generator = build_generator()
discriminator = build_discriminator()

# =============================
# LOSS & OPTIMIZER
# =============================
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

def train_step(images):
    noise = tf.random.normal([images.shape[0], NOISE_DIM])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(fake, training=True)
        gen_loss = loss_fn(tf.ones_like(fake_out), fake_out)
        disc_loss = loss_fn(tf.ones_like(real_out), real_out) + \
                    loss_fn(tf.zeros_like(fake_out), fake_out)

    gen_opt.apply_gradients(zip(gt.gradient(gen_loss, generator.trainable_variables),
                                generator.trainable_variables))
    disc_opt.apply_gradients(zip(dt.gradient(disc_loss, discriminator.trainable_variables),
                                 discriminator.trainable_variables))

# =============================
# TRAIN LOOP WITH TQDM
# =============================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(dataset):
        train_step(batch)
