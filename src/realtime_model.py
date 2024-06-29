import numpy as np
import cv2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model('eqn-detect-model.h5')

label_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '=', 17: 'x', 18: 'y', 19: 'z',
    # Add more mappings as needed
}

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to match the input shape expected by the model
    resized_image = cv2.resize(gray_image, (256, 64))  # Adjust these dimensions as needed
    
    # Normalize the pixel values
    normalized_image = resized_image.astype('float32') / 255.0
    
    # Add the batch dimension and channel dimension
    processed_image = np.expand_dims(normalized_image, axis=(0, -1))
    
    return processed_image
def label_to_text(label):
    if label in label_to_char:
        return label_to_char[label]
    else:
        return ''
    
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = 100, 200, 200, 100  # Change these values as needed
    equation_region = frame[y:y+h, x:x+w]
    preprocessed = preprocess_image(equation_region)

    predictions = model.predict(np.array([equation_region]))
    predicted_label = np.argmax(predictions, axis=1)[0]

    recognized_equation = ''.join([label_to_text(label) for label in predicted_label])

    cv2.putText(frame, f'Solution: {recognized_equation}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()