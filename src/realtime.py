import cv2
import pytesseract
from sympy import sympify

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = 100, 200, 200, 100  # Change these values as needed
    equation_region = frame[y:y+h, x:x+w]
    equation_text = pytesseract.image_to_string(equation_region)

    try:
        # draw a rectangle around the equation region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        equation = sympify(equation_text.strip())
        solution = equation.evalf()
        #when solution = 2.0, take a screenshot
        cv2.putText(frame, f'Solution: {solution}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, 'No equation detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()