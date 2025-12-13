"""
AEE Protocol v0.2.0 - Torture Test con Llama 3.1 Local
================================================================

Este test VERDADERAMENTE valida si el watermark sobrevive
a una reescritura por IA (Llama 3.1 local).

FLUJO:
1. Generar texto original
2. Obtener embedding del texto (con watermark)
3. Hacer que Llama lo reescriba
4. Obtener embedding del texto reescrito
5. Verificar si watermark sobrevive

PREREQUISITO:
- LM Studio corriendo con Llama 3.1 en localhost:1234
- pip install ollama sentencepiece

INSTALACIÓN:
pip install ollama sentence-transformers
================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import time

# Importar watermark
from src.core.watermark import AEEWatermarkV2

# Intentar importar ollama (para comunicarse con LM Studio)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  ollama no instalado. Instala con: pip install ollama")

# Para embeddings locales (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers no instalado. Instala con: pip install sentence-transformers")


class AEETortureTest:
    """
    Torture test del AEE Protocol contra reescrituras de IA.
    """
    
    def __init__(self,
                 watermark: AEEWatermarkV2,
                 model_name: str = "meta-llama/Llama-2-7b-chat",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Args:
            watermark: Instancia de AEEWatermarkV2
            model_name: Nombre del modelo en Ollama/LM Studio
            embedding_model: Modelo para generar embeddings
            ollama_base_url: URL base de Ollama (LM Studio)
        """
        self.watermark = watermark
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        # Cargar modelo de embeddings
        if EMBEDDINGS_AVAILABLE:
            print(f"📥 Cargando modelo de embeddings: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ImportError("sentence-transformers es requerido")
        
        # Test results
        self.results = {
            'total_tests': 0,
            'watermarks_survived': 0,
            'watermarks_lost': 0,
            'average_confidence': 0.0,
            'tests': []
        }
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding para texto usando modelo local.
        
        Args:
            text: Texto a vectorizar
            
        Returns:
            Embedding normalizado
        """
        embedding = self.embedding_model.encode(text)
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype('float32')
    
    def rewrite_with_llama(self, text: str, instruction: str = "Reescribe el siguiente texto de forma más clara y concisa:") -> str:
        """
        Usa Llama (vía Ollama/LM Studio) para reescribir texto.
        
        Args:
            text: Texto a reescribir
            instruction: Instrucción para Llama
            
        Returns:
            Texto reescrito
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama es requerido para este test")
        
        prompt = f"{instruction}\n\n{text}"
        
        try:
            # Llamar a Ollama (que se conecta a LM Studio)
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.7,
                    'num_predict': 200
                }
            )
            
            rewritten = response['response'].strip()
            return rewritten
        
        except Exception as e:
            print(f"❌ Error conectando con Llama: {e}")
            print(f"   ¿LM Studio está corriendo en {self.ollama_base_url}?")
            raise
    
    def run_torture_test(self,
                        test_texts: list = None,
                        num_iterations: int = 3) -> Dict[str, Any]:
        """
        Ejecuta torture test contra reescrituras de Llama.
        
        Args:
            test_texts: Lista de textos a probar
            num_iterations: Cuántas reescrituras probar por texto
            
        Returns:
            Resultados detallados
        """
        if test_texts is None:
            test_texts = [
                "El watermarking vectorial es una técnica criptográfica para proteger embeddings de IA.",
                "Los embeddings son representaciones numéricas de texto en espacios vectoriales de alta dimensión.",
                "La trazabilidad de datos es crítica para auditorías de modelos de lenguaje."
            ]
        
        print(f"\n🔥 INICIANDO TORTURE TEST")
        print(f"   Textos: {len(test_texts)}")
        print(f"   Iteraciones por texto: {num_iterations}")
        print(f"   Total de tests: {len(test_texts) * num_iterations}\n")
        
        for text_idx, original_text in enumerate(test_texts, 1):
            print(f"\n{'='*60}")
            print(f"TEXTO {text_idx}: {original_text[:50]}...")
            print(f"{'='*60}")
            
            # Step 1: Embedding original
            print(f"\n[1/4] Generando embedding original...")
            original_embedding = self.get_embedding(original_text)
            print(f"      ✓ Embedding: {original_embedding.shape}")
            
            # Step 2: Watermark
            print(f"[2/4] Inyectando watermark...")
            marked_embedding, metadata = self.watermark.inject(original_embedding)
            print(f"      ✓ Watermark inyectado")
            print(f"      ✓ Confianza inicial: {self.watermark.detect(marked_embedding)['confidence']:.1%}")
            
            # Step 3: Reescrituras
            for iter_num in range(1, num_iterations + 1):
                print(f"\n   [ITERACIÓN {iter_num}/{num_iterations}]")
                
                try:
                    # Reescribir con Llama
                    print(f"   [3/4] Reescribiendo con Llama...")
                    instructions = [
                        "Reescribe de forma más clara:",
                        "Parafrasea manteniendo el significado:",
                        "Expresa la misma idea de otra forma:"
                    ]
                    instruction = instructions[(iter_num - 1) % len(instructions)]
                    
                    rewritten_text = self.rewrite_with_llama(
                        original_text,
                        instruction=instruction
                    )
                    print(f"       Original: {original_text[:40]}...")
                    print(f"       Reescrito: {rewritten_text[:40]}...")
                    
                    # Embedding reescrito
                    print(f"   [4/4] Obteniendo embedding reescrito...")
                    rewritten_embedding = self.get_embedding(rewritten_text)
                    
                    # Detectar watermark
                    detection = self.watermark.detect(rewritten_embedding)
                    
                    # Registrar resultado
                    test_result = {
                        'text_idx': text_idx,
                        'iteration': iter_num,
                        'original_text': original_text,
                        'rewritten_text': rewritten_text,
                        'detected': detection['detected'],
                        'confidence': detection['confidence'],
                        'similarity': detection['similarity'],
                        'threshold': detection['threshold'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    self.results['tests'].append(test_result)
                    self.results['total_tests'] += 1
                    
                    if detection['detected']:
                        self.results['watermarks_survived'] += 1
                        status = "✅ SOBREVIVIÓ"
                    else:
                        self.results['watermarks_lost'] += 1
                        status = "❌ PERDIDO"
                    
                    print(f"       {status}")
                    print(f"       Confianza: {detection['confidence']:.1%}")
                    print(f"       Similitud: {detection['similarity']:.4f}")
                
                except Exception as e:
                    print(f"       ❌ Error: {e}")
        
        # Calcular estadísticas
        if self.results['total_tests'] > 0:
            confidences = [t['confidence'] for t in self.results['tests']]
            self.results['average_confidence'] = np.mean(confidences)
            self.results['survival_rate'] = (
                self.results['watermarks_survived'] / self.results['total_tests']
            )
        
        return self._print_summary()
    
    def _print_summary(self) -> Dict[str, Any]:
        """Imprime resumen de resultados."""
        print(f"\n\n{'='*60}")
        print(f"📊 RESUMEN DE TORTURE TEST")
        print(f"{'='*60}")
        print(f"Total de tests: {self.results['total_tests']}")
        print(f"Watermarks sobrevivieron: {self.results['watermarks_survived']}")
        print(f"Watermarks perdidos: {self.results['watermarks_lost']}")
        print(f"Tasa de supervivencia: {self.results['survival_rate']:.1%}")
        print(f"Confianza promedio: {self.results['average_confidence']:.1%}")
        
        if self.results['survival_rate'] > 0.8:
            print(f"\n✅ RESULTADO: PROTOCOLO ROBUSTO")
        elif self.results['survival_rate'] > 0.5:
            print(f"\n⚠️  RESULTADO: PROTOCOLO MODERADO")
        else:
            print(f"\n❌ RESULTADO: PROTOCOLO DÉBIL")
        
        print(f"\n💾 Guardando resultados...")
        
        # Guardar resultados en JSON
        output_file = f"torture_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Resultados guardados en: {output_file}")
        
        return self.results


# =====================================================
# EJECUCIÓN
# =====================================================

if __name__ == "__main__":
    print("🔒 AEE Protocol v0.2.0 - Torture Test Local\n")
    
    if not OLLAMA_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        print("❌ DEPENDENCIAS FALTANTES")
        print("\nInstala con:")
        print("  pip install ollama sentence-transformers")
        sys.exit(1)
    
    # Inicializar watermark
    watermark = AEEWatermarkV2(
        user_id=35664619,
        strength=0.25,
        embedding_dim=384  # MiniLM usa 384 dimensiones
    )
    
    # Inicializar torture test
    torture_test = AEETortureTest(
        watermark=watermark,
        model_name="llama2",  # O el nombre que tengas en LM Studio
        embedding_model="all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434"
    )
    
    # Textos de prueba
    test_texts = [
        "El watermarking vectorial protege la propiedad intelectual en modelos de IA.",
        "Los embeddings son representaciones de alta dimensión del significado semántico.",
        "La trazabilidad de datos es fundamental para auditorías de cumplimiento."
    ]
    
    # Ejecutar torture test
    try:
        results = torture_test.run_torture_test(
            test_texts=test_texts,
            num_iterations=2
        )
    except Exception as e:
        print(f"\n❌ Error durante torture test: {e}")
        print("\n⚠️  CHECKLIST:")
        print("   1. ¿LM Studio está abierto?")
        print("   2. ¿Llama está cargado en LM Studio?")
        print("   3. ¿Está escuchando en http://localhost:11434?")
        print("   4. Intenta: curl http://localhost:11434/api/generate")