# De Novo Design of RNA G-Quadruplex Binding Peptides

**Author:** Subhajit Dutta  
**Institution:** Department of Biochemistry and Molecular Cell Biology (IBMZ), University Medical Center Hamburg-Eppendorf, Germany  

This repository contains the official code and computational pipeline for the manuscript:  
> *"De Novo Design of RNA G-Quadruplex Binding Peptides via Cross-Modal Variational Autoencoder"*.

## Pipeline Overview
This pipeline leverages the ESM-2 protein language model and a Variational Autoencoder (VAE) to learn the biochemical latent space of natural RNA G-quadruplex (RG4) binding proteins. It generates synthetic intrinsically disordered peptides (IDPs) that specifically target RG4 structures through Arginine groove insertion and Tyrosine aromatic capping.

## Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YourUsername/RG4-VAE-Peptide-Design.git](https://github.com/YourUsername/RG4-VAE-Peptide-Design.git)
cd RG4-VAE-Peptide-Design
pip install -r requirements.txt
