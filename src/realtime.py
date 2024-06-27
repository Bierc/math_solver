import cv2
import pytesseract
from sympy import sympify

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = 100, 100, 200, 50  # Change these values as needed
    equation_region = frame[y:y+h, x:x+w]
    equation_text = pytesseract.image_to_string(equation_region)

    try:
        equation = sympify(equation_text.strip())
        solution = equation.evalf()
        print(f'{equation} = {solution}')
    except Exception as e:
        print("impossible to solve the equation")

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()