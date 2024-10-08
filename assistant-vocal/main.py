import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Charger le modèle et le processeur Wav2Vec2
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Fonction pour reconnaître la parole
def recognize_speech():
    print("Enregistrement... (parlez maintenant)")
    recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1)  # Changer ici à 16000
    sd.wait()  # Attendre la fin de l'enregistrement
    audio = recording.flatten()

    # Convertir l'audio en entrée pour le modèle
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)  # Changer ici à 16000
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Prendre le mot prédit avec la probabilité la plus élevée
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


# Fonction pour exécuter des commandes
def execute_command(command):
    if "éteins-toi" in command.lower():
        print("Commande reçue : Éteindre la machine.")
        # Code pour éteindre la machine (commenté pour la sécurité)
        # os.system("shutdown /s /t 1") # À décommenter sur votre machine
    elif "applaudir" in command.lower():
        print("Geste détecté : Applaudissements !")
    else:
        print("Commande non reconnue.")

# Boucle principale
while True:
    transcription = recognize_speech()
    print(f"Vous avez dit : {transcription}")
    execute_command(transcription)
