from setuptools import setup, find_packages

setup(
    name="aee-protocol",
    version="8.0.0",
    author="Franco Carricondo",
    author_email="tu-email@ejemplo.com",  # Pon tu email real si quieres
    description="Protocolo de trazabilidad e identidad para embeddings vectoriales",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ProtocoloAEE/aee-protocol",
    packages=find_packages(),  # Esto busca automáticamente la carpeta 'aeeprotocol'
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        # Aquí puedes agregar 'pinecone-client' u otros si decides hacerlos obligatorios
    ],
)