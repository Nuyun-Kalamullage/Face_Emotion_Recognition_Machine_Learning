import tensorflow as tf

# Convert the model

#load h5 module
model=tf.keras.models.load_model('facialemotionmodel.h5')
converter = tf.lite.TFLiteConverter.from_saved_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)