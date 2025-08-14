import tensorflow as tf

model = tf.keras.models.load_model("best_asl_model.h5")
model.summary()
