"""
AEE Protocol - SDK Client
User-friendly interface for watermarking operations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from ..core.engine import AEEMathEngine


class AEEClient:
    """
    Main client for AEE Protocol operations.
    
    Provides simple interface for:
    - Watermark injection
    - Watermark detection/verification
    - Configuration management
    """
    
    def __init__(self, user_id: int, strength: float = 0.5, target_fpr: float = 0.02):
        """
        Initialize AEE client.
        
        Args:
            user_id: Unique identifier for watermark (keep secret!)
            strength: Watermark strength (0.3-0.7 recommended)
            target_fpr: Target false positive rate (e.g., 0.02 for 2%)
        """
        self.user_id = int(user_id)
        self.engine = AEEMathEngine(strength=strength, target_fpr=target_fpr)
        
        print(f"🔐 AEE Client initialized:")
        print(f"   User ID: {self.user_id}")
        print(f"   Strength: {strength}")
        print(f"   Target FPR: {target_fpr:.2%}")
        print(f"   Threshold: {self.engine.threshold:.6f}")
    
    def watermark(self, embedding: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Inject watermark into embedding.
        
        Args:
            embedding: Input vector (any dimension, will be treated as 768D)
            
        Returns:
            Tuple of (watermarked_vector, metadata)
        """
        # Ensure correct shape and type
        if embedding.ndim != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")
        
        # Pad or truncate to 768 if needed
        if len(embedding) != 768:
            print(f"⚠️  Input dimension {len(embedding)} != 768. Adjusting...")
            embedding = self._adjust_dimension(embedding)
        
        # Inject watermark
        watermarked, metadata = self.engine.inject(embedding, self.user_id)
        
        return watermarked, metadata
    
    def verify(self, embedding: np.ndarray, 
               original_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Verify if embedding contains user's watermark.
        
        Args:
            embedding: Vector to test
            original_embedding: Optional original for reference
            
        Returns:
            Dictionary with verification results
        """
        # Adjust dimension if needed
        if len(embedding) != 768:
            embedding = self._adjust_dimension(embedding)
        
        if original_embedding is not None and len(original_embedding) != 768:
            original_embedding = self._adjust_dimension(original_embedding)
        
        # Detect watermark
        result = self.engine.detect(
            embedding, 
            self.user_id, 
            original_embedding
        )
        
        # Rename keys for better user experience
        return {
            'verified': result['detected'],
            'confidence_score': result['correlation_score'],
            'confidence_level': result.get('confidence', 0.0),
            'threshold': result['threshold'],
            'strength': result['strength'],
            'target_fpr': result['target_fpr'],
            'user_id': result['user_id'],
            'metadata': {
                'algorithm': 'AEEv8.3',
                'correlation': result['correlation_score'],
                'threshold': result['threshold'],
                'margin': result['correlation_score'] - result['threshold'],
            }
        }
    
    def _adjust_dimension(self, embedding: np.ndarray, target_dim: int = 768) -> np.ndarray:
        """
        Adjust embedding dimension to target size.
        
        Args:
            embedding: Input vector
            target_dim: Target dimension (default 768)
            
        Returns:
            Adjusted vector of target_dim
        """
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        if current_dim > target_dim:
            # Truncate
            return embedding[:target_dim].copy()
        else:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=embedding.dtype)
            padded[:current_dim] = embedding
            return padded
    
    def batch_watermark(self, embeddings: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Watermark multiple embeddings.
        
        Args:
            embeddings: 2D array of shape (n_embeddings, dimension)
            
        Returns:
            Tuple of (watermarked_embeddings, metadata_list)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Batch embeddings must be 2D, got shape {embeddings.shape}")
        
        n_embeddings = embeddings.shape[0]
        watermarked = np.zeros_like(embeddings)
        metadata_list = []
        
        for i in range(n_embeddings):
            watermarked[i], metadata = self.watermark(embeddings[i])
            metadata_list.append(metadata)
        
        return watermarked, metadata_list
    
    def batch_verify(self, embeddings: np.ndarray) -> list:
        """
        Verify multiple embeddings.
        
        Args:
            embeddings: 2D array of shape (n_embeddings, dimension)
            
        Returns:
            List of verification results
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Batch embeddings must be 2D, got shape {embeddings.shape}")
        
        n_embeddings = embeddings.shape[0]
        results = []
        
        for i in range(n_embeddings):
            result = self.verify(embeddings[i])
            results.append(result)
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary with client configuration
        """
        return {
            'user_id': self.user_id,
            'strength': self.engine.strength,
            'target_fpr': self.engine.target_fpr,
            'threshold': self.engine.threshold,
            'algorithm': 'AEEv8.3',
        }


# Example usage function
def example_usage():
    """Show example usage of the AEE client."""
    print("\n" + "="*60)
    print("AEE CLIENT - EXAMPLE USAGE")
    print("="*60)
    
    # 1. Initialize client
    client = AEEClient(user_id=123456, strength=0.5, target_fpr=0.02)
    
    # 2. Create a sample embedding
    original = np.random.randn(768).astype(np.float32)
    original = original / np.linalg.norm(original)
    
    print(f"\n📊 Original embedding:")
    print(f"   Shape: {original.shape}")
    print(f"   Norm: {np.linalg.norm(original):.6f}")
    
    # 3. Watermark it
    watermarked, metadata = client.watermark(original)
    
    print(f"\n💧 Watermarked embedding:")
    print(f"   Norm: {np.linalg.norm(watermarked):.6f}")
    print(f"   Similarity to original: {np.dot(original, watermarked):.6f}")
    print(f"   Metadata hash: {metadata.get('watermarked_hash', 'N/A')}")
    
    # 4. Verify it
    result = client.verify(watermarked)
    
    print(f"\n🔍 Verification result:")
    print(f"   Verified: {result['verified']}")
    print(f"   Score: {result['confidence_score']:.6f}")
    print(f"   Threshold: {result['threshold']:.6f}")
    print(f"   Confidence: {result['confidence_level']:.3f}")
    
    # 5. Test with random vector (should fail)
    random_vec = np.random.randn(768).astype(np.float32)
    random_vec = random_vec / np.linalg.norm(random_vec)
    random_result = client.verify(random_vec)
    
    print(f"\n🎲 Random vector test:")
    print(f"   Verified: {random_result['verified']}")
    print(f"   Score: {random_result['confidence_score']:.6f}")
    
    print("="*60)
    print("✅ Example completed")


if __name__ == "__main__":
    example_usage()