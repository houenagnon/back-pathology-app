from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Chargement du modèle YOLO
model = YOLO('./best.pt').to('cpu')  # Assurez-vous d'utiliser 'cpu' si 'cuda' n'est pas disponible

# Récupération des noms des classes du modèle
class_names = model.names  # YOLO stocke les noms des classes ici

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Lecture de l'image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Exécution de la détection
    results = model(img)

    # Si `results` est une liste contenant les détections
    if results and isinstance(results, list) and len(results) > 0:
        detections = []
        for result in results[0].boxes:  # Parcourir les boîtes détectées
            bbox = result.xyxy.tolist()  # Coordonnées du rectangle délimitant
            conf = result.conf.tolist()  # Confiance de la détection
            class_id = result.cls.tolist()  # ID de la classe détectée

            # Dans certains cas, `class_id` pourrait être une liste, donc on itère
            for idx in class_id:
                class_name = class_names[int(idx)]  # Récupérer le nom de la classe
                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": idx,
                    "class_name": class_name
                })

        return {'detections': detections}
    else:
        return {'message': 'Aucune détection trouvée'}
