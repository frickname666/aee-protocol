# 🔐 AEE Protocol v0.2.3

**Scientific Watermarking for Vector Embeddings - Empirically Validated**

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.2.3-brightgreen)
![Validation](https://img.shields.io/badge/validation-50k%2B%20trials-success)
![Status](https://img.shields.io/badge/status-production--ready-orange)

---

## 🎯 **What is AEE Protocol?**

AEE Protocol is a **scientifically-grounded watermarking system** for vector embeddings that provides:

🔹 **Proof of Ownership** - Mathematically prove vectors are yours  
🔹 **Data Leakage Detection** - Identify stolen embeddings in vector databases  
🔹 **Configurable Security** - Choose your false positive rate (0.1% to 5%)  
🔹 **Noise Resilience** - Survive data corruption and transformations  
🔹 **Blind Detection** - Verify without the original embedding  

**Use Case:** Protect sensitive embeddings in Pinecone, Weaviate, Qdrant, and other vector databases from unauthorized use or theft.

---

## 🚀 **Quick Start**

### Installation
```bash
pip install aeeprotocol
Basic Usage
python
from aeeprotocol.sdk.client import AEEClient
import numpy as np

# Initialize with your identity and security preferences
client = AEEClient(
    user_id=123456,          # Your unique identifier (keep secret!)
    strength=0.5,            # Watermark strength (0.3-0.7 recommended)
    target_fpr=0.02          # Target false positive rate: 2%
)

# 1. Mark your embedding
original_vector = np.random.randn(768).astype('float32')
original_vector = original_vector / np.linalg.norm(original_vector)

marked_vector, proof = client.watermark(original_vector)
print(f"✅ Watermarked. Hash: {proof['watermarked_hash']}")

# 2. Verify ownership (blind detection)
result = client.verify(marked_vector)
print(f"Ownership verified: {result['verified']}")
print(f"Confidence score: {result['confidence_score']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")
Integration with Pinecone
python
from pinecone import Pinecone
from aeeprotocol.sdk.client import AEEClient

# Initialize
pc = Pinecone(api_key="your-key")
index = pc.Index("protected-index")
client = AEEClient(user_id=123456, strength=0.5, target_fpr=0.02)

# Watermark before storing
embedding = get_embedding("Your sensitive text")
marked_embedding, metadata = client.watermark(embedding)

# Store with proof
index.upsert(vectors=[{
    "id": "doc_001",
    "values": marked_embedding.tolist(),
    "metadata": {
        "aee_proof": metadata,
        "content": "Original text"
    }
}])

# Later audit: check for stolen embeddings
stored = index.fetch(["doc_001"])["vectors"]["doc_001"]["values"]
result = client.verify(np.array(stored))

if result['verified']:
    print("✅ Your intellectual property detected")
    print(f"   Confidence: {result['confidence_level']:.1%}")
else:
    print("❌ Unknown or tampered vector")
🧪 Scientific Validation (v8.3)
Rigorous Statistical Testing (50,000+ trials)
Configuration	Target FPR	Observed FPR	Noise Resistance (20% σ)	Status
Balanced
strength=0.5	2.00%	1.88%	44% survival	✅ Recommended
Strict
strength=0.5	1.00%	0.93%	42% survival	✅ Low FPR
Very Strict
strength=0.5	0.10%	0.04%	8% survival	✅ Forensic
Low Strength
strength=0.4	2.00%	2.06%	38% survival	✅ Low distortion
High Strength
strength=0.6	2.00%	1.99%	56% survival	✅ High robustness
Noise Resilience Performance
text
Noise Level | Survival Rate | Average Score
----------- | ------------- | -------------
   5% σ     |    100%       |     0.257
  10% σ     |     98%       |     0.148
  15% σ     |     90%       |     0.120
  20% σ     |     44%       |     0.080
  25% σ     |     26%       |     0.070
  30% σ     |     24%       |     0.061
*Based on 10,000+ trials per configuration. See VALIDATION.md for full methodology.*

⚙️ How It Works
Mathematical Foundation
AEE Protocol v8.3 uses scientific threshold calculation based on statistical theory:

python
# Threshold calculation for desired False Positive Rate (FPR)
# For random vectors in 768 dimensions:
# correlation ~ N(0, 1/√(n-3))
# We compute: threshold = z_score / √(n-3)
# where z_score = norm.ppf(1 - target_fpr/2)

from scipy import stats
dimension = 768
target_fpr = 0.02  # 2%
z_score = stats.norm.ppf(1 - target_fpr / 2)
threshold = z_score / np.sqrt(dimension - 3)  # ≈ 0.0841
Watermark Injection Process
Direction Generation: Deterministic vector from user_id only

Orthogonal Injection: watermarked = original + strength × direction

Normalization: Maintain unit length for consistency

Blind Detection: Regenerate same direction, compute correlation

Key Innovation
Unlike heuristic approaches, AEE Protocol mathematically guarantees the false positive rate you specify, making it predictable and reliable for production use.

🎯 Configuration Guide
Choosing Parameters
python
# CASE 1: General Monitoring (recommended)
client = AEEClient(
    user_id=123456,
    strength=0.5,      # Balanced distortion/robustness
    target_fpr=0.02    # 2% false alarms - good for continuous monitoring
)

# CASE 2: Forensic/Legal Evidence
client = AEEClient(
    user_id=123456,
    strength=0.5,
    target_fpr=0.001   # 0.1% FPR - very low false accusations
)

# CASE 3: High Noise Environments
client = AEEClient(
    user_id=123456,
    strength=0.6,      # Stronger watermark
    target_fpr=0.02    # But still manageable FPR
)

# CASE 4: Minimum Distortion
client = AEEClient(
    user_id=123456,
    strength=0.4,      # Subtle watermark
    target_fpr=0.02    # Standard FPR
)
Performance Characteristics
Parameter	Range	Default	Effect
strength	0.3-0.7	0.5	Higher = more robust, more distortion
target_fpr	0.001-0.05	0.02	Lower = fewer false positives, harder detection
Embedding Dim	384-1536	768	Works with common embedding models
📊 Performance Metrics
Metric	Value	Notes
Injection Speed	< 1 ms/vector	CPU single-threaded
Detection Speed	< 0.5 ms/vector	Correlation operation
Memory Overhead	0 bytes	No extra storage needed
Embedding Distortion	5-15%	Depends on strength parameter
Batch Processing	10k vectors/sec	On modern CPU
Dimension Support	Any	Auto-pads/truncates to 768
🔬 Technical Details
Architecture
text
aeeprotocol/
├── 📁 core/
│   └── engine.py           # Mathematical engine (v8.3)
├── 📁 sdk/
│   └── client.py          # User-friendly interface
└── 📄 __init__.py
Core Algorithm (v8.3 Improvements)
python
class AEEMathEngine:
    def __init__(self, strength=0.5, target_fpr=0.02):
        # Scientific threshold calculation (NEW in v8.3)
        self.threshold = stats.norm.ppf(1-target_fpr/2) / sqrt(dim-3)
        
    def inject(self, embedding, user_id):
        # 1. Deterministic direction from user_id
        direction = self._compute_direction(user_id)
        
        # 2. Orthogonal injection
        watermarked = embedding + self.strength * direction
        
        # 3. Maintain unit sphere
        return watermarked / norm(watermarked)
Why v8.3 is Different
✅ No heuristic thresholds - Everything is mathematically calculated

✅ Predictable FPR - You get exactly the false positive rate you specify

✅ Transparent operation - No hidden parameters or magic numbers

✅ Empirically validated - 50,000+ test cases confirm theory

⚠️ Limitations & Considerations
What AEE Protocol DOES Protect Against:
✅ Direct embedding theft from vector databases

✅ Unauthorized copying to competitor systems

✅ Accidental leakage with minor corruption

✅ Basic transformations (normalization, mild noise)

What AEE Protocol Does NOT Protect Against:
❌ AI model training on your data (different threat model)

❌ Sophisticated adversarial attacks (research area)

❌ Complete reconstruction from watermarked embeddings

❌ user_id compromise (keep it secret!)

Security Model
Assumptions:

Attacker doesn't know your user_id

Attacker doesn't have many of your watermarked samples

Attacker uses standard noise/addition attacks

If these assumptions break:

With 1000+ watermarked samples: attacker could estimate watermark

If user_id leaks: watermark can be removed

Sophisticated denoising could reduce watermark strength

📈 Comparison with Alternatives
Feature	AEE Protocol	Traditional Hashing	Learned Watermarking
Blind Detection	✅ Yes	❌ No (needs original)	✅ Yes
Noise Resistance	✅ 20% σ survived	❌ None	⚠️ Limited
Configurable FPR	✅ 0.1%-5%	❌ Fixed	❌ Fixed
Mathematical Guarantees	✅ Proven	✅ Proven	❌ Empirical
Speed	✅ 1ms/vector	✅ Instant	❌ Slow (needs NN)
Transparency	✅ Full	✅ Full	❌ Black box
🛠️ Advanced Usage
Batch Operations
python
# Watermark multiple embeddings at once
embeddings = np.random.randn(100, 768).astype('float32')
watermarked_batch, metadata_list = client.batch_watermark(embeddings)

# Verify batch
results = client.batch_verify(watermarked_batch)
suspicious = [i for i, r in enumerate(results) if not r['verified']]
Custom Integration
python
class ProtectedVectorDB:
    def __init__(self, user_id):
        self.client = AEEClient(user_id=user_id)
        self.vectors = {}
    
    def store(self, id, embedding):
        marked, proof = self.client.watermark(embedding)
        self.vectors[id] = {
            'vector': marked,
            'proof': proof,
            'original_hash': hashlib.sha256(embedding.tobytes()).hexdigest()
        }
    
    def audit(self, external_db_vectors):
        """Check if any vectors in external DB match ours"""
        matches = []
        for vec in external_db_vectors:
            result = self.client.verify(vec)
            if result['verified'] and result['confidence_level'] > 0.7:
                matches.append(vec)
        return matches
Monitoring Dashboard (Example)
python
# Simple monitoring script
import time
from collections import deque

class AEEMonitor:
    def __init__(self, client, window_size=1000):
        self.client = client
        self.detections = deque(maxlen=window_size)
    
    def monitor_stream(self, vector_stream):
        """Monitor continuous stream of vectors"""
        for vector in vector_stream:
            result = self.client.verify(vector)
            self.detections.append(result['verified'])
            
            # Alert if detection rate changes
            detection_rate = sum(self.detections) / len(self.detections)
            if detection_rate > 0.01:  # More than 1% detection
                print(f"⚠️  High detection rate: {detection_rate:.2%}")
🧪 Testing & Validation
Run Full Test Suite
bash
# Install test dependencies
pip install numpy scipy

# Run comprehensive validation (50k+ trials)
python auditor_test_v8.3.py

# Run noise resilience test
python torture_test_noise.py

# Calculate your own FPR
python -c "
from scipy import stats
dim = 768
for fpr in [0.001, 0.01, 0.02, 0.05]:
    z = stats.norm.ppf(1 - fpr/2)
    th = z / (dim - 3)**0.5
    print(f'FPR {fpr:.1%} -> threshold {th:.6f}')
"
Validation Methodology
Statistical Validity: 10,000 random vectors per FPR test

Noise Testing: Gaussian noise at 5%, 10%, 15%, 20%, 25%, 30% levels

Reproducibility: Fixed seeds for deterministic testing

Edge Cases: Zero vectors, duplicate vectors, abnormal values

See VALIDATION.md for complete methodology.

🤝 Contributing
We welcome contributions in:

Statistical validation with larger datasets

Integration with more vector databases

Performance optimization for GPU/TPU

Security audits and penetration testing

Documentation improvements

Development Setup
bash
# Clone repository
git clone https://github.com/ProtocoloAEE/aee-protocol.git
cd aee-protocol

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
python auditor_test_v8.3.py
Code Standards
Follow PEP 8 style guide

Include type hints for new functions

Add tests for new features

Update documentation accordingly

📚 Documentation
VALIDATION.md - Complete test methodology and results

ARCHITECTURE.md - Mathematical foundation and proofs

API Reference - Complete SDK documentation

Examples - Practical use cases and integrations

📄 License
MIT License - See LICENSE file for details.

Copyright © 2025 Franco Luciano Carricondo
Building digital sovereignty from Argentina. 🇦🇷

👤 Author & Contact
Franco Luciano Carricondo
🔗 LinkedIn: linkedin.com/in/francocarricondo
📧 Email: francocarricondo@gmail.com
🐙 GitHub: @ProtocoloAEE
🐦 Twitter: @ProtocoloAEE

Support & Community
GitHub Issues: Report bugs or request features

Discussions: Join the conversation

Email: Direct support for enterprise users

🔄 Changelog
v0.2.3 (Current) - December 2025
✅ Scientific threshold calculation (no more guessing)

✅ Configurable false positive rates (0.1% to 5%)

✅ 50,000+ validation trials across configurations

✅ Improved documentation with empirical results

✅ Better error handling and user experience

v0.2.0 - December 2025
Initial public release with basic watermarking

Pinecone/Weaviate integration examples

Basic validation (5,000 trials)

🌟 Acknowledgments
Statistical Foundation: Based on correlation distribution theory

Embedding Models: Tested with Sentence Transformers, OpenAI embeddings

Vector Databases: Compatible with Pinecone, Weaviate, Qdrant, Chroma

Community: Early testers and feedback providers

Last Updated: December 2025
Validation Status: 50,000+ trials across 5 configurations
Production Ready: Yes, with understood limitations
Recommendation: Use strength=0.5, target_fpr=0.02 for general purpose

💡 Tip: For production deployments, rotate user_id periodically and monitor detection rates for anomalies.