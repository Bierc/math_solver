import cv2
import numpy as np
import tensorflow as tf
import sympy as sp
from tensorflow.keras.models import load_model

# Carregar o modelo e definir as classes
class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'times']
model = load_model('C:\\UTFPR\\math_solver\\src\\eqn-detect-model.h5')

# Função para binarizar a imagem
def binarize(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_binary_img = cv2.bitwise_not(binarized)
    expanded_img = np.expand_dims(inverted_binary_img, -1)
    return expanded_img

# Função para detectar contornos e retornar as bounding boxes
def getOverlap(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start)

import cv2

import cv2
import numpy as np

def detect_contours(frame, min_width=20, min_height=20, canny_threshold1=100, canny_threshold2=150):
    # Converter para escala de cinza
    input_image_cpy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar o detector de bordas de Canny
    edges = cv2.Canny(input_image_cpy, canny_threshold1, canny_threshold2)
    
    # Encontrar contornos
    contours_list, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for i, c in enumerate(contours_list):
        x, y, w, h = cv2.boundingRect(c)
        
        # Filtrar contornos com base no tamanho mínimo
        if w >= min_width and h >= min_height:
            bounding_boxes.append([x, y, w, h, i])  # Adicionar índice do contorno
    
    filtered_boxes = []
    for i, box1 in enumerate(bounding_boxes):
        x1, y1, w1, h1, idx1 = box1
        is_within_another = False
        
        # Verificar se este contorno está dentro de outro contorno maior
        current_hierarchy = hierarchy[0][idx1]
        parent_idx = current_hierarchy[3]
        
        while parent_idx != -1:
            px, py, pw, ph = cv2.boundingRect(contours_list[parent_idx])
            if pw >= min_width and ph >= min_height:
                is_within_another = True
                break
            parent_idx = hierarchy[0][parent_idx][3]
        
        if not is_within_another:
            # Verificar sobreposição com outros contornos
            for j, box2 in enumerate(bounding_boxes):
                if i != j:
                    x2, y2, w2, h2, idx2 = box2
                    # Verificar se há sobreposição significativa
                    if (x1 < x2 + w2 and x1 + w1 > x2 and
                        y1 < y2 + h2 and y1 + h1 > y2):
                        # Se o contorno atual for menor, não adicioná-lo
                        if w1 * h1 < w2 * h2:
                            is_within_another = True
                            break
            
            if not is_within_another:
                filtered_boxes.append([x1, y1, w1, h1])
    
    return filtered_boxes


# Função para redimensionar e preencher a imagem
def resize_pad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w / h
    if aspect > 1:
        new_w = sw
        new_h = int(round(new_w / aspect))
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = int(round(new_h * aspect))
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled_img

# Função para pré-processar a imagem da webcam
def preprocess_webcam_image(img):
    inverted_binary_img = binarize(img)
    keep = detect_contours(img)
    eqn_list = []
    for (x, y, w, h) in sorted(keep, key=lambda x: x[0]):
        img_crop = inverted_binary_img[y:y+h, x:x+w]
        img_resized = resize_pad(img_crop, (45, 45), 0)
        img_expanded = np.expand_dims(img_resized, axis=0)
        img_expanded = np.expand_dims(img_expanded, axis=-1)
        pred_class = class_names[np.argmax(model.predict(img_expanded))]
        if pred_class == "times":
            pred_class = "*"
        eqn_list.append(pred_class)
    eqn = "".join(eqn_list)
    return eqn

def solve_equation(equation):
    # Trata casos de existência de igualdade sem valor
    if "=" in equation:
        # Dividir a equação em lado esquerdo e direito
        left, right = equation.split("=")
        left = left.strip()
        right = right.strip()
        if not left and not right:
            raise("Não tem equação")
        if not left:
            equation = right
        elif not right:
            equation = left
        else:
            equation = equation

    # Detectar o caso baseado no conteúdo da equação
    if isinstance(equation, list):
        left_expr = equation[0].lhs
        right_expr = equation[0].rhs
    else:
        if "=" in str(equation):
            left_expr = equation.lhs
            right_expr = equation.rhs
        else:
            left_expr = equation
            right_expr = sp.S.Zero

    # Tentar converter as expressões para símbolos
    try:
        left_expr = sp.sympify(left_expr)
        right_expr = sp.sympify(right_expr)
    except sp.SympifyError:
        return "Equação inválida"

    # Caso 1: equação apenas numérica
    if left_expr.is_number and right_expr.is_number:
        result = left_expr - right_expr
        return result.evalf()  # Usar evalf() para obter o valor numérico

    # Caso 2 e 3: equações com variáveis
    variables = list(left_expr.free_symbols | right_expr.free_symbols)

    # Resolver a equação usando solve
    solutions = sp.solve(left_expr - right_expr, variables)

    # Melhorar a legibilidade das soluções
    numeric_solutions = [sol.evalf() for sol in solutions]
    
    return numeric_solutions

def most_frequent(List):
    return max(set(List), key = List.count)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
else:
    count = 0 
    eqn_list = []
    result = "Nao foi encontrada uma equacao"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        eqn = preprocess_webcam_image(frame)
        eqn_list.append(eqn)
        # Process frame to detect contours
        detected_boxes = detect_contours(frame)

        for box in detected_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle


        if count % 20 == 19:
            print(eqn_list)
            real_eqn = most_frequent(eqn_list)
            print(real_eqn)
            if real_eqn:
                result = solve_equation(real_eqn)
            else:
                result = "Nao foi encontrada uma equacao"
            eqn_list = []
        
        # Display the frame with detected contours
        if result == "Nao foi encontrada uma equacao":
            cv2.putText(frame, f"{result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{real_eqn} = {result:.5}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        count += 1
        cv2.imshow('Detected Contours', frame)
    

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
