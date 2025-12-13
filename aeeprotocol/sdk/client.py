import numpy as np
from typing import Optional, Dict, Tuple, Any
from ..core.engine import AEEMathEngine

class AEEClient:
    """
    Cliente de Alto Nivel para el Protocolo AEE.
    Maneja la conversión de tipos y la gestión del motor.
    """

    def __init__(self, user_id: int, strength: float = 0.25):
        """
        Args:
            user_id: Identificador único (DNI/Passport/UUID)
            strength: Intensidad de la marca (0.1 - 0.5)
        """
        self.user_id = user_id
        self.engine = AEEMathEngine(strength=strength)

    def watermark(self, vector: Any, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Inyecta marca de agua. Acepta listas de Python o Numpy Arrays.
        """
        # Conversión automática a Numpy
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
            
        watermarked_vec, proof_meta = self.engine.inject(vector, self.user_id)
        
        if metadata:
            proof_meta.update(metadata)
            
        return watermarked_vec, proof_meta

    def verify(self, vector: Any) -> Dict[str, Any]:
        """
        Verifica la propiedad del vector.
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        score = self.engine.detect(vector, self.user_id)
        is_detected = score > self.engine.threshold

        return {
            "verified": bool(is_detected),
            "confidence_score": round(float(score), 4),
            "threshold": float(self.engine.threshold),
            "owner_id": self.user_id
        }
         from .client import AEEClient