import numpy as np
import cv2
import tensorflow as tf

# Configurar o uso da GPU no TensorFlow, se disponível

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Carregar o modelo Keras salvo
model = tf.keras.models.load_model('eqn-detect-model.h5')

# Mostrar a arquitetura do modelo para verificar a entrada esperada
model.summary()

# Dicionário de mapeamento de rótulos para caracteres
label_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '=', 17: 'x', 18: 'y', 19: 'z',
    # Adicione mais mapeamentos conforme necessário
}

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (45, 45))
    normalized_image = resized_image.astype('float32') / 255.0
    processed_image = np.expand_dims(normalized_image, axis=-1)  # Adicionar dimensão do canal (escala de cinza)
    processed_image = np.expand_dims(processed_image, axis=0)  # Adicionar dimensão do lote
    return processed_image

def label_to_text(label):
    if label in label_to_char:
        return label_to_char[label]
    else:
        return ''
    
cap = cv2.VideoCapture(0)
batch_size = 8
while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = 100, 200, 200, 100  # Altere esses valores conforme necessário
    equation_region = frame[y:y+h, x:x+w]
    # preprocessed = preprocess_image(equation_region)

    processed_frame = preprocess_image(frame)

    # predictions = model.predict(preprocessed)
    predictions = model.predict(processed_frame, batch_size=batch_size)
    predicted_label = np.argmax(predictions, axis=1)[0]

    recognized_equation = label_to_text(predicted_label)
    print(predictions)
    print(predicted_label)
    print(recognized_equation)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f'Solution: {recognized_equation}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()