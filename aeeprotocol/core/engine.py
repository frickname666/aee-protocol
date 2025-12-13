import numpy as np
import hashlib
from typing import Tuple, Dict, Optional

class AEEMathEngine:
    """
    Motor matemático puro del Protocolo AEE (Architecture for Embedding Evidence).
    Realiza la inyección y detección de marcas de agua usando subespacios ortogonales.
    """
    
    def __init__(self, strength: float = 0.25):
        self.strength = strength
        # El umbral por defecto es el 40% de la fuerza de inyección
        self.threshold = strength * 0.4

    def compute_direction(self, embedding: np.ndarray, user_id: int) -> np.ndarray:
        """
        Genera un vector de dirección determinista único para el usuario y el contenido.
        """
        # 1. Semilla criptográfica: Hash(VectorBytes + UserID)
        seed_bytes = embedding.tobytes() + str(user_id).encode()
        # Usamos SHA256 para máxima entropía y tomamos los primeros 32 bits
        seed_int = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16) % (2**32)
        
        # 2. Generación Pseudo-Aleatoria Aislada
        rng = np.random.RandomState(seed_int)
        direction = rng.randn(len(embedding))
        
        # 3. Ortogonalización (Gram-Schmidt)
        # Esto asegura que la marca no altere la semántica principal del vector
        emb_norm = embedding / np.linalg.norm(embedding)
        direction = direction - np.dot(direction, emb_norm) * emb_norm
        
        # 4. Normalización final de la dirección
        return direction / np.linalg.norm(direction)

    def inject(self, embedding: np.ndarray, user_id: int) -> Tuple[np.ndarray, dict]:
        """Aplica la marca de agua matemática."""
        # Asegurar normalización del input
        norm = np.linalg.norm(embedding)
        base_emb = embedding / norm if abs(norm - 1.0) > 0.01 else embedding
        
        # Calcular dirección secreta
        direction = self.compute_direction(base_emb, user_id)
        
        # Inyección: V_marcado = V + (fuerza * Dirección)
        watermarked = base_emb + self.strength * direction
        
        meta = {
            'algo': 'AEEv8',
            'strength': self.strength,
            'hash': hashlib.sha256(watermarked.tobytes()).hexdigest()[:8]
        }
        return watermarked, meta

    def detect(self, embedding: np.ndarray, user_id: int) -> float:
        """Calcula el score de similitud (Producto Punto) con la marca."""
        embedding_norm = embedding / np.linalg.norm(embedding)
        direction = self.compute_direction(embedding_norm, user_id)
        
        # El producto punto nos dice qué tan alineado está el vector con la marca
        return float(np.dot(embedding_norm, direction))