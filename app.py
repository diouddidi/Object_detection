import cv2

def detect_objects_realtime(cascade_path):
    # Charger le classificateur en cascade
    cascade = cv2.CascadeClassifier(cascade_path)
    
    # Capturer la vidéo depuis la webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Real-Time Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Exemple d'utilisation
detect_objects_realtime("haarcascade_frontalface_default.xml")