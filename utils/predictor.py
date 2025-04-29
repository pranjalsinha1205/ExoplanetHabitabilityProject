import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load model components once
model_bundle = joblib.load('earth_similarity_model.joblib')
kmeans = model_bundle['kmeans']
scaler = model_bundle['scaler']
earth_center = model_bundle['earth_cluster_center']
features = model_bundle['features']

# Earth-like planet values (scaled)
earth_scaled = scaler.transform(np.array([[365.25, 1.0, 1.0, 1.0, 0.0167, 1.0, 288.0, 5778, 1.0, 1.0]]))

def predict_similarity(user_input: dict) -> tuple:
    """
    Accepts a dict of feature values and returns:
    - similarity % to Earth cluster
    - is_habitable (True/False)
    """
    # Convert dict to scaled array
    values = np.array([[user_input[feat] for feat in features]])
    scaled_values = scaler.transform(values)

    # Euclidean distance from Earth's scaled vector
    dist = np.linalg.norm(scaled_values - earth_scaled)

    # Earth's similarity always 100% (for reference)
    max_dist = np.linalg.norm(earth_scaled - earth_center)

    # Compute similarity as percentage (scaled)
    similarity = max(0, 1 - dist / max_dist) * 100

    # Threshold adjusted for stricter habitability check
    is_habitable = similarity >= 90  # You can adjust this threshold as needed

    return round(similarity, 2), is_habitable
