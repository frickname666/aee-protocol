"""
AEE Protocol - Core Mathematical Engine
v8.3: Scientific threshold calculation for configurable FPR
"""

import numpy as np
import hashlib
from typing import Tuple, Dict, Optional
import sys

# Try to import scipy for statistical functions
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Using approximate threshold calculation.")


class AEEMathEngine:
    """
    Core mathematical engine for AEE Protocol.
    
    Features:
    - Scientific threshold calculation for target FPR
    - Deterministic direction generation from user_id only
    - Consistent normalization
    - Configurable strength and false positive rate
    """
    
    # Default embedding dimension (commonly 768 for sentence transformers)
    DEFAULT_DIM = 768
    
    def __init__(self, strength: float = 0.5, target_fpr: float = 0.02):
        """
        Initialize the engine with scientific parameters.
        
        Args:
            strength: Watermark strength (0.3-0.7 recommended)
                     Higher = more robust, but more distortion
            target_fpr: Target false positive rate (e.g., 0.02 for 2%)
                       Lower = fewer false alarms, but harder detection
        """
        if not 0.1 <= strength <= 1.0:
            raise ValueError(f"strength must be between 0.1 and 1.0, got {strength}")
        
        if not 1e-6 <= target_fpr <= 0.5:
            raise ValueError(f"target_fpr must be between 1e-6 and 0.5, got {target_fpr}")
        
        self.strength = float(strength)
        self.target_fpr = float(target_fpr)
        
        # Calculate threshold scientifically
        self.threshold = self._calculate_threshold(target_fpr)
        
        # Cache for directions (key: user_id)
        self._direction_cache = {}
        
        # Debug info
        self._print_config()
    
    def _calculate_threshold(self, target_fpr: float) -> float:
        """
        Calculate detection threshold for desired false positive rate.
        
        For random vectors in high dimensions, the correlation follows:
        corr ~ N(0, 1/√(n-3)) approximately
        
        We want: P(|corr| > threshold) = target_fpr
        => threshold = z_score / √(n-3)
        where z_score is the (1 - target_fpr/2) quantile of N(0,1)
        """
        n = self.DEFAULT_DIM
        
        if HAS_SCIPY:
            # Exact calculation using scipy
            z_score = stats.norm.ppf(1 - target_fpr / 2)
        else:
            # Approximate using error function inverse
            # Approximation for z-score: sqrt(2) * erfinv(1 - target_fpr)
            import math
            from scipy.special import erfcinv
            z_score = math.sqrt(2) * erfcinv(target_fpr)
        
        threshold = z_score / np.sqrt(n - 3)
        
        # Ensure threshold is reasonable (correlation ∈ [0,1])
        return float(np.clip(threshold, 0.01, 0.5))
    
    def _print_config(self):
        """Print configuration summary for debugging."""
        print("\n" + "="*50)
        print("AEE MATH ENGINE v8.3 - CONFIGURATION")
        print("="*50)
        print(f"✓ Strength:          {self.strength:.3f}")
        print(f"✓ Target FPR:        {self.target_fpr:.3%}")
        print(f"✓ Calculated threshold: {self.threshold:.6f}")
        print(f"✓ Using scipy:       {HAS_SCIPY}")
        print("="*50)
    
    def _compute_direction(self, user_id: int) -> np.ndarray:
        """
        Generate deterministic direction vector from user_id.
        
        The direction depends ONLY on user_id, not the embedding.
        This enables blind detection.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Normalized direction vector (norm = 1.0)
        """
        # Check cache first
        if user_id in self._direction_cache:
            return self._direction_cache[user_id]
        
        # Create deterministic seed from user_id
        seed_bytes = str(user_id).encode('utf-8')
        seed_hash = hashlib.sha256(seed_bytes).digest()
        
        # Use first 8 bytes for 64-bit seed
        seed_int = int.from_bytes(seed_hash[:8], byteorder='big', signed=False)
        
        # Generate direction with deterministic randomness
        rng = np.random.RandomState(seed_int % (2**32))
        direction = rng.randn(self.DEFAULT_DIM).astype(np.float32)
        
        # Exact normalization
        norm = np.linalg.norm(direction)
        if norm < 1e-10:  # Avoid division by zero
            direction = np.ones(self.DEFAULT_DIM, dtype=np.float32)
            norm = np.linalg.norm(direction)
        
        direction = direction / norm
        
        # Verify normalization
        verified_norm = np.linalg.norm(direction)
        if abs(verified_norm - 1.0) > 1e-6:
            # Renormalize if needed
            direction = direction / verified_norm
        
        # Cache for future use
        self._direction_cache[user_id] = direction
        
        return direction
    
    def inject(self, embedding: np.ndarray, user_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Inject watermark into embedding vector.
        
        Args:
            embedding: Input vector (will be normalized if not unit length)
            user_id: User identifier for watermark
            
        Returns:
            Tuple of (watermarked_vector, metadata_dict)
        """
        # Ensure float32 precision
        embedding = embedding.astype(np.float32)
        
        # Normalize input if needed
        input_norm = np.linalg.norm(embedding)
        if abs(input_norm - 1.0) > 1e-6:
            embedding = embedding / input_norm
        
        # Get deterministic direction
        direction = self._compute_direction(user_id)
        
        # Inject watermark: W = V + α * D
        watermarked = embedding + self.strength * direction
        
        # Normalize output (important for consistency)
        watermarked = watermarked / np.linalg.norm(watermarked)
        
        # Create metadata
        metadata = {
            'user_id': user_id,
            'strength': self.strength,
            'target_fpr': self.target_fpr,
            'threshold': self.threshold,
            'embedding_norm': float(np.linalg.norm(embedding)),
            'watermarked_norm': float(np.linalg.norm(watermarked)),
            'embedding_hash': hashlib.sha256(embedding.tobytes()).hexdigest()[:12],
            'watermarked_hash': hashlib.sha256(watermarked.tobytes()).hexdigest()[:12],
            'algorithm': 'AEEv8.3',
            'timestamp': np.datetime64('now').astype(str),
        }
        
        return watermarked.astype(np.float32), metadata
    
    def detect(self, embedding: np.ndarray, user_id: int, 
               original_embedding: Optional[np.ndarray] = None) -> Dict:
        """
        Detect watermark in embedding vector.
        
        Args:
            embedding: Vector to test (will be normalized)
            user_id: User identifier to test against
            original_embedding: Optional original for reference (not used in blind detection)
            
        Returns:
            Dictionary with detection results and confidence metrics
        """
        # Normalize test vector
        test_vector = embedding.astype(np.float32).copy()
        test_norm = np.linalg.norm(test_vector)
        
        if abs(test_norm - 1.0) > 1e-6:
            test_vector = test_vector / test_norm
        
        # Get same direction used for injection
        direction = self._compute_direction(user_id)
        
        # Calculate correlation (absolute value for detection)
        correlation = np.abs(np.dot(test_vector, direction))
        
        # Detection decision
        detected = correlation > self.threshold
        
        # Calculate confidence metrics
        result = {
            'detected': bool(detected),
            'correlation_score': float(correlation),
            'threshold': float(self.threshold),
            'strength': float(self.strength),
            'target_fpr': float(self.target_fpr),
            'user_id': user_id,
        }
        
        # Add confidence level
        if detected:
            # Normalized confidence: (score - threshold) / (max_possible - threshold)
            max_possible = 1.0  # Maximum possible correlation
            confidence = (correlation - self.threshold) / (max_possible - self.threshold)
            result['confidence'] = float(np.clip(confidence, 0.0, 1.0))
        else:
            result['confidence'] = 0.0
        
        # If original is provided, calculate similarity
        if original_embedding is not None:
            original_norm = original_embedding / np.linalg.norm(original_embedding)
            similarity = float(np.dot(test_vector, original_norm))
            result['original_similarity'] = similarity
        
        return result
    
    def verify(self, embedding: np.ndarray, user_id: int) -> Dict:
        """
        Alias for detect() with cleaner naming for SDK use.
        """
        return self.detect(embedding, user_id)


# Helper function for quick testing
def test_engine():
    """Quick test of the engine functionality."""
    print("\n🧪 QUICK ENGINE TEST")
    print("="*50)
    
    # Create engine with 2% FPR target
    engine = AEEMathEngine(strength=0.5, target_fpr=0.02)
    
    # Generate random vector
    original = np.random.randn(768).astype(np.float32)
    original = original / np.linalg.norm(original)
    
    # Inject watermark
    watermarked, metadata = engine.inject(original, user_id=12345)
    
    print(f"Original norm: {np.linalg.norm(original):.6f}")
    print(f"Watermarked norm: {np.linalg.norm(watermarked):.6f}")
    print(f"Similarity: {np.dot(original, watermarked):.6f}")
    
    # Detect watermark
    result = engine.detect(watermarked, user_id=12345)
    
    print(f"\nDetection result:")
    print(f"  Detected: {result['detected']}")
    print(f"  Score: {result['correlation_score']:.6f}")
    print(f"  Threshold: {result['threshold']:.6f}")
    print(f"  Confidence: {result.get('confidence', 0):.3f}")
    
    # Test with random vector (should not detect)
    random_vec = np.random.randn(768).astype(np.float32)
    random_vec = random_vec / np.linalg.norm(random_vec)
    random_result = engine.detect(random_vec, user_id=12345)
    
    print(f"\nRandom vector test:")
    print(f"  Detected: {random_result['detected']}")
    print(f"  Score: {random_result['correlation_score']:.6f}")
    
    print("="*50)
    print("✅ Engine test completed")
    
    return engine, result


if __name__ == "__main__":
    # Run test if module is executed directly
    test_engine()