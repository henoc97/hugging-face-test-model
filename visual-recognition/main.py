import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import mediapipe as mp

# Initialiser MediaPipe pour la détection de pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Charger le modèle et le processeur de détection d'objets
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

def classify_action(landmarks):
    """Classifie les actions basées sur les landmarks de pose."""
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]:
        return "Vous levez les bras"
    elif landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] > landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]:
        return "Vous êtes assis"
    return "Action non reconnue"

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image pour MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Détection d'objets
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Récupérer les résultats de détection d'objets
    target_sizes = torch.tensor([frame.shape[:2]])
    results_detection = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Dessiner les résultats de détection d'objets sur l'image
    for score, label, box in zip(results_detection["scores"], results_detection["labels"], results_detection["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", 
                    (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Traitement des landmarks de pose
    if results.pose_landmarks:
        landmarks = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
        
        # Classifier l'action en fonction des landmarks
        action = classify_action(landmarks)
        cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image avec les détections
    cv2.imshow("Détection d'Objets et de Poses", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
