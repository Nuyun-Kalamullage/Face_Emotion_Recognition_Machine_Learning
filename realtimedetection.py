import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

# Load the model architecture and weights
with open('facialemotionmodel.json', 'r') as json_file:
    model_json = json_file.read()

# Ensure custom objects are registered properly
custom_objects = {
    'Sequential': tf.keras.Sequential,
}

# Load the model using a custom object scope
model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
model.load_weights('facialemotionmodel.h5')
# Save the model without the optimizer state
model.save('facialemotionmodellite.h5', include_optimizer=False)


# Define the labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to 48x48
    resized_frame = cv2.resize(gray_frame, (48, 48))
    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0
    # Reshape the frame for prediction
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))

    # Predict the emotion
    prediction = model.predict(reshaped_frame)
    emotion = emotion_labels[np.argmax(prediction)]

    # Display the emotion on the frame
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
