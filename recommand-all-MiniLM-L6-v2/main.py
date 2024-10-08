from sentence_transformers import SentenceTransformer, util
import torch

# Charger un modèle pré-entraîné pour la similarité sémantique
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Exemples de descriptions de produits
products = [
    "Je cherche chaussures en cuir",
    " Il s'agit d'un t-shirt en coton de haute qualité, parfait pour une tenue décontractée",
    " Une chemise en coton confortable pour les activités quotidiennes ",
    " Cette élégante robe de soirée est parfaite pour les événements formels ",
    " Chaussures de course très confortables pour les courses de longue durée ",
    " Des chaussures en cuir pour les réunions d'affaires et les événements"
]

# Encoder les descriptions de produits en vecteurs d'embeddings
embeddings = model.encode(products, convert_to_tensor=True)

# Calculer la similarité entre les produits (cosine similarity)
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

# Fonction pour recommander des produits similaires
def recommend_products(product_id, top_n=3):
    # Obtenir les scores de similarité pour le produit donné
    product_similarities = similarity_matrix[product_id]
    
    # Obtenir les indices des produits les plus similaires
    similar_products = torch.topk(product_similarities, k=top_n + 1)  # +1 pour ignorer le produit lui-même
    recommended_ids = similar_products.indices[1:]  # On exclut l'identifiant du produit lui-même
    
    print(f"Recommendations for product: '{products[product_id]}'")
    for idx in recommended_ids:
        print(f" - {products[idx]}")
        
# Exemple : Recommander des produits similaires au premier produit
recommend_products(0)
