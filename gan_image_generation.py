"""
AI-Based Image Generation using GAN (Updated with Live Output)
Dataset: MNIST
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =============================
# CONFIGURATION (CPU FRIENDLY)
# =============================
BUFFER_SIZE = 60000
BATCH_SIZE = 128        # Reduced for CPU
EPOCHS = 20             # Reduce epochs for faster output
NOISE_DIM = 100
NUM_EXAMPLES = 16

os.makedirs("generated_images", exist_ok=True)

# =============================
# LOAD & PREPROCESS DATASET
# =============================
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print("Dataset loaded successfully")

# =============================
# GENERATOR MODEL
# =============================
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2),
                               padding="same", use_bias=False, activation="tanh")
    ])
    return model

# =============================
# DISCRIMINATOR MODEL
# =============================
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding="same",
                      input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

print("Models initialized")

# =============================
# LOSS & OPTIMIZERS
# =============================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# =============================
# TRAIN STEP (NO @tf.function)
# =============================
def train_step(images):
    noise = tf.random.normal([images.shape[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# =============================
# SAVE GENERATED IMAGES
# =============================
def save_images(epoch):
    noise = tf.random.normal([NUM_EXAMPLES, NOISE_DIM])
    generated_images = generator(noise, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(NUM_EXAMPLES):
        plt.subplot(4,4,i+1)
        plt.imshow((generated_images[i] + 1) / 2, cmap="gray")
        plt.axis("off")

    plt.savefig(f"generated_images/image_epoch_{epoch}.png")
    plt.close()

# =============================
# TRAINING LOOP (LIVE OUTPUT)
# =============================
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"\n🚀 Starting Epoch {epoch+1}/{epochs}")
        for step, image_batch in enumerate(dataset):
            train_step(image_batch)
            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step}")
        save_images(epoch + 1)
        print(f"✅ Epoch {epoch+1}/{epochs} completed")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    print("Starting GAN training...\n")
    train(dataset, EPOCHS)
    print("\n🎉 Training complete! Check generated_images folder.")
