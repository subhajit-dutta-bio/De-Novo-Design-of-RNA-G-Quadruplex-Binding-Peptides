#!/usr/bin/env python3
"""
==============================================================================
De Novo Design of RNA G-Quadruplex Binding Peptides via Cross-Modal Variational Autoencoder
==============================================================================
Author: Subhajit Dutta*
Affiliation: Department of Biochemistry and Molecular Cell Biology (IBMZ), 
             Center for Experimental Medicine, University Medical Center 
             Hamburg-Eppendorf, Hamburg, Germany
Contact: s.dutta@uke.de ; dsubhajit.edu@gmail.com

Description:
This script contains the complete computational pipeline for generating synthetic 
RNA G-quadruplex (RG4) binding peptides using a cross-modal Variational Autoencoder 
(VAE) trained on ESM-2 protein language model embeddings.

Pipeline Phases:
    1. data_acquisition   : Process natural RG4 binding proteins into fragments
    2. extract_embeddings : Generate 1280-D ESM-2 embeddings
    3. train_vae          : Train the VAE and generate novel sequences
    4. filter_candidates  : High-Throughput Virtual Screening (HTVS)
    5. prep_af3           : Generate JSON configurations for AlphaFold 3
    6. generate_figures   : Create publication-quality figures
==============================================================================
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# Global Configuration & Hyperparameters
# ==============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "processed_peptides"
EMBEDDING_DIR = "esm2_embeddings"
FIG_DIR = "figures"

# Fragment constraints
MIN_LENGTH = 50
MAX_LENGTH = 2500
WINDOW_SIZE = 50
STRIDE = 25

# VAE Hyperparameters
EMBED_DIM = 1280
SEQ_LEN = 50
LATENT_DIM = 128
HIDDEN_DIM = 512
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# ==============================================================================
# PHASE 1: Data Acquisition
# ==============================================================================
def data_acquisition():
    print("--- Phase 1: Data Acquisition ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_fragments = []
    total_positives = 0

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please add FASTA files.")
        return

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith((".fasta", ".fa")):
            file_path = os.path.join(DATA_DIR, file_name)
            for record in SeqIO.parse(file_path, "fasta"):
                seq_len = len(record.seq)
                if MIN_LENGTH <= seq_len <= MAX_LENGTH and record.id.startswith("POS_"):
                    total_positives += 1
                    seq_str = str(record.seq)
                    for i in range(0, len(seq_str) - WINDOW_SIZE + 1, STRIDE):
                        fragment_seq = seq_str[i:i+WINDOW_SIZE]
                        if len(fragment_seq) == WINDOW_SIZE:
                            all_fragments.append({
                                "Source_Protein_ID": record.id,
                                "Fragment_Sequence": fragment_seq,
                            })

    print(f"Total POSITIVE RG4BPs extracted: {total_positives}")
    print(f"Total positive peptide fragments generated: {len(all_fragments)}")

    df = pd.DataFrame(all_fragments)
    output_csv = os.path.join(OUTPUT_DIR, "G4_binding_peptides_CLEAN.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved clean data to: {output_csv}\n")

# ==============================================================================
# PHASE 2: ESM-2 Feature Extraction
# ==============================================================================
def extract_embeddings():
    print("--- Phase 2: ESM-2 Feature Extraction ---")
    try:
        import esm
    except ImportError:
        raise ImportError("Please install fair-esm: pip install fair-esm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    input_csv = os.path.join(OUTPUT_DIR, "G4_binding_peptides_CLEAN.csv")
    df = pd.read_csv(input_csv)
    sequences = df["Fragment_Sequence"].tolist()
    
    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    batch_size = 64
    all_embeddings = []

    print("Extracting 1280-dimensional embeddings...")
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for j, seq_len in enumerate((batch_tokens != alphabet.padding_idx).sum(1)):
                seq_rep = token_representations[j, 1:seq_len - 1].mean(0)
                all_embeddings.append(seq_rep.cpu())

    embeddings_tensor = torch.stack(all_embeddings)
    output_path = os.path.join(EMBEDDING_DIR, "g4_positive_embeddings.pt")
    torch.save(embeddings_tensor, output_path)
    print(f"Saved embeddings to: {output_path}\n")

# ==============================================================================
# PHASE 3: VAE Architecture & Training
# ==============================================================================
class G4PeptideDataset(Dataset):
    def __init__(self, csv_file, pt_file):
        self.df = pd.read_csv(csv_file)
        self.embeddings = torch.load(pt_file)
        self.seqs = torch.tensor([[AA_TO_IDX.get(aa, 0) for aa in seq] 
                                  for seq in self.df['Fragment_Sequence']], dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.seqs[idx]

class ESM2_to_Seq_VAE(nn.Module):
    def __init__(self):
        super(ESM2_to_Seq_VAE, self).__init__()
        self.fc1 = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, SEQ_LEN * VOCAB_SIZE)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.fc4(F.relu(self.fc3(z))).view(-1, SEQ_LEN, VOCAB_SIZE)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_seq, target_seq, mu, logvar):
    BCE = F.cross_entropy(recon_seq.view(-1, VOCAB_SIZE), target_seq.view(-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(num_generate=1000000):
    print("--- Phase 3: VAE Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = os.path.join(OUTPUT_DIR, "G4_binding_peptides_CLEAN.csv")
    pt_file = os.path.join(EMBEDDING_DIR, "g4_positive_embeddings.pt")
    
    dataset = G4PeptideDataset(csv_file, pt_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ESM2_to_Seq_VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for emb, seq in dataloader:
            emb, seq = emb.to(device), seq.to(device)
            optimizer.zero_grad()
            recon_seq, mu, logvar = model(emb)
            loss = vae_loss_function(recon_seq, seq, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss / len(dataset):.4f}")

    # Generation
    model.eval()
    print(f"\nGenerating {num_generate} novel candidate sequences...")
    synthetic_peptides = []
    with torch.no_grad():
        # Batch generation to prevent OOM
        batch_gen = 10000
        for _ in range(0, num_generate, batch_gen):
            z = torch.randn(batch_gen, LATENT_DIM).to(device)
            generated_indices = torch.argmax(model.decode(z), dim=-1).cpu().numpy()
            for seq_idx in generated_indices:
                synthetic_peptides.append("".join([IDX_TO_AA[idx] for idx in seq_idx]))

    df_out = pd.DataFrame({"Synthetic_Fragment": synthetic_peptides})
    out_file = f"Generated_G4_Peptides_{num_generate}.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved generated sequences to '{out_file}'\n")

# ==============================================================================
# PHASE 4: HTVS Filtering
# ==============================================================================
def filter_candidates(input_file="Generated_G4_Peptides_1000000.csv"):
    print("--- Phase 4: Heuristic HTVS Filtering ---")
    df = pd.read_csv(input_file)
    valid_candidates = []

    for seq in df['Synthetic_Fragment'].tolist():
        rgg_count = seq.count('RGG')
        aromatic_count = seq.count('Y') + seq.count('F')
        enrichment_score = sum([seq.count(aa) for aa in ['G', 'S', 'Y', 'F', 'R']]) / len(seq)

        if rgg_count >= 1 and aromatic_count >= 3 and enrichment_score >= 0.70:
            valid_candidates.append({
                'Sequence': seq,
                'RGG_Motif_Count': rgg_count,
                'Aromatic_Count': aromatic_count,
                'Enrichment_Score': round(enrichment_score, 3)
            })

    df_candidates = pd.DataFrame(valid_candidates)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            by=['Enrichment_Score', 'RGG_Motif_Count', 'Aromatic_Count'],
            ascending=[False, False, False]
        )
        print(f"Found {len(df_candidates)} ELITE G4-binding candidates.")
        df_candidates.head(100).to_csv('Top_100_Elite_G4_Peptides.csv', index=False)
        print("Saved elite candidates to 'Top_100_Elite_G4_Peptides.csv'\n")
    else:
        print("No sequences passed the strict filters.\n")

# ==============================================================================
# PHASE 5: AlphaFold 3 Preparation
# ==============================================================================
def prep_af3():
    print("--- Phase 5: Generating AF3 JSON Configs ---")
    def create_json(filename, job_name, pep_seq, target_seq, target_type="rnaSequence"):
        config = [{"name": job_name, "sequences": [
            {"proteinChain": {"sequence": pep_seq, "count": 1}},
            {target_type: {"sequence": target_seq, "count": 1}},
            {"ion": {"ion": "K", "count": 2}}
        ]}]
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    g4_rna, g4_dna = "UGGAGGCUGGA", "TGGAGGCTGGA"
    seq_1_wt = "RGGGRGGRGGGRGRGRGRGRRGGYRGGGGGGGRGGYGGRYRGRGRRGGGG"
    seq_neg = "LVGNKADIRDTAATEGQKCVPGHFGEKLAMTYGALFCETSAKDGSNIVEA"

    os.makedirs("af3_configs", exist_ok=True)
    create_json("af3_configs/AF3_Elite_Cand1_RNA.json", "Elite_Cand1_RNA", seq_1_wt, g4_rna)
    create_json("af3_configs/AF3_Elite_Cand1_DNA.json", "Elite_Cand1_DNA", seq_1_wt, g4_dna, "dnaSequence")
    create_json("af3_configs/AF3_Negative_Control.json", "Negative_Peptide_G4", seq_neg, g4_rna)
    print("AF3 JSON files generated in 'af3_configs/' directory.\n")

# ==============================================================================
# PHASE 6: Figure Generation (Matplotlib/Seaborn)
# ==============================================================================
def generate_figures():
    print("--- Phase 6: Publication Figure Generation ---")
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import logomaker
    except ImportError:
        print("Please install logomaker: pip install logomaker")
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 14, 'figure.dpi': 300, 'savefig.bbox': 'tight'})

    # AA Composition
    df_nat = pd.read_csv(os.path.join(OUTPUT_DIR, "G4_binding_peptides_CLEAN.csv"))
    df_syn = pd.read_csv("Top_100_Elite_G4_Peptides.csv")
    
    def get_freq(seq_list):
        counter = Counter("".join(seq_list))
        tot = sum(counter.values())
        return {aa: (c/tot)*100 for aa, c in counter.items()}

    nat_freq = get_freq(df_nat['Fragment_Sequence'].dropna().tolist())
    syn_freq = get_freq(df_syn['Sequence'].tolist())

    df_comp = pd.DataFrame({
        'Amino Acid': AMINO_ACIDS * 2,
        'Frequency (%)': [nat_freq.get(aa, 0) for aa in AMINO_ACIDS] + [syn_freq.get(aa, 0) for aa in AMINO_ACIDS],
        'Source': ['Natural Binders'] * 20 + ['Elite Synthetic'] * 20
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_comp, x='Amino Acid', y='Frequency (%)', hue='Source', ax=ax)
    ax.set_title("Amino Acid Composition Enrichment")
    fig.savefig(os.path.join(FIG_DIR, "Fig2A_AA_Composition.pdf"))
    
    # Sequence Logo
    fig, ax = plt.subplots(figsize=(12, 3))
    logo = logomaker.Logo(logomaker.alignment_to_matrix(df_syn['Sequence'].tolist()), ax=ax)
    ax.set_title("Sequence Grammar of Elite Synthetic Peptides")
    fig.savefig(os.path.join(FIG_DIR, "Fig2B_Sequence_Logo.pdf"))
    
    print("Figures saved in 'figures/' directory.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De Novo RG4 Peptide Designer Pipeline")
    parser.add_argument("--phase", type=str, required=True, 
                        choices=["data_acquisition", "extract_embeddings", "train_vae", 
                                 "filter_candidates", "prep_af3", "generate_figures", "all"],
                        help="Select the pipeline phase to execute.")
    args = parser.parse_args()

    if args.phase in ["data_acquisition", "all"]: data_acquisition()
    if args.phase in ["extract_embeddings", "all"]: extract_embeddings()
    if args.phase in ["train_vae", "all"]: train_vae(num_generate=1000000)
    if args.phase in ["filter_candidates", "all"]: filter_candidates()
    if args.phase in ["prep_af3", "all"]: prep_af3()
    if args.phase in ["generate_figures", "all"]: generate_figures()
