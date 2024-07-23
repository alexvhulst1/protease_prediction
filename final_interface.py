import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import streamlit as st
import re
import ijson
from transformers import T5Tokenizer, T5EncoderModel
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################

## E3 Ligase Interaction Finder ##

dic_e3 = {}
with open('E3/dic_E3.json') as f:
    dic_e3 = json.load(f)

dic_go_e3 = {}
with open('E3/dic_GO_problem_2.json') as f:
    dic_go_e3 = json.load(f)

inv_dic_enzyme_e3 = {v: k for k, v in dic_e3.items()}

class model_embedding(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(model_embedding, self).__init__()
        self.fc = nn.Linear(embed_dim, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        
    def forward(self, embed):
        x = F.relu(self.fc(embed))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    
class model_embedding_go(nn.Module):
    def __init__(self, embed_dim, go_dim, output_dim):
        super(model_embedding_go, self).__init__()
        self.fc_embed = nn.Linear(embed_dim, 2048)
        self.fc_go = nn.Linear(go_dim, 2048)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def forward(self, embed, go):
        x_embed = F.relu(self.fc_embed(embed))
        x_go = F.relu(self.fc_go(go))
        x = torch.cat((x_embed, x_go), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

model_e3_without_go = model_embedding(1024, 185)
model_e3_with_go = model_embedding_go(1024, 2000, 185)

model_e3_without_go.load_state_dict(torch.load('E3/model_embedding_problem_2_final.pt'))
model_e3_with_go.load_state_dict(torch.load('E3/model_embedding_go_problem_2_final.pt'))

# Charger les modèles T5 pour les embeddings de séquences
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
model_prott5 = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')

# Function to find E3 interactions from the model
def find_e3_from_sequence(sequence, number_outputs = 3, swissprot = '?', go = []):
    sequence_examples = [sequence]
    sequence_Example = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    ids = tokenizer.batch_encode_plus(
        sequence_Example, add_special_tokens=True, padding="longest"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        model_prott5.to(device)

        with torch.no_grad():
            embeddings = model_prott5(
                input_ids=tokenized_sequences, attention_mask=attention_mask
            )

        # Extract embeddings for the first sequence in the batch while removing padded & special tokens
        emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

        # Derive a single representation (per-protein embedding) for the whole protein
        emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

        embedding = emb_0_per_protein  # shape (1024)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory, switching to CPU")
            device = 'cpu'
            tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            model_prott5.to(device)

            with torch.no_grad():
                embeddings = model_prott5(
                    input_ids=tokenized_sequences, attention_mask=attention_mask
                )

            # Extract embeddings for the first sequence in the batch while removing padded & special tokens
            emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

            # Derive a single representation (per-protein embedding) for the whole protein
            emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

            embedding = emb_0_per_protein  # shape (1024)
        else:
            raise e
    seq = sequence

    embed = embedding.unsqueeze(0)

    embed = embed.to(device)

    if swissprot == True:
        model_e3_with_go.to(device)
        go_int = [0 for i in range(2000)]
        for i in range(len(go)):
            if go[i] in dic_go_e3.keys():
                go_int[dic_go_e3[go[i]]] = 1
        go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
        
        outputs = model_e3_with_go(embed, go_tensor)

    if swissprot == False:
        model_e3_without_go.to(device)
        outputs = model_e3_without_go(embed)
    
    if swissprot == '?':
        go = go_from_seq(sequence)
        if go == []:
            model_e3_without_go.to(device)
            outputs = model_e3_without_go(embed)
        else:
            model_e3_with_go.to(device)
            go_int = [0 for i in range(2000)]
            for i in range(len(go)):
                if go[i] in dic_go_e3.keys():
                    go_int[dic_go_e3[go[i]]] = 1
            go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model_e3_with_go(embed, go_tensor)  

    top_indices = np.argsort(outputs[0].cpu().detach().numpy())[-number_outputs:][::-1]
    results = []

    for idx in top_indices:
        category = inv_dic_enzyme_e3[idx]
        probability = outputs[0][idx]
        results.append({"Category": category, "Probability": round(probability.item(), 4)})

    return results

def find_e3(uniprot_acc, number_outputs=3):
    found = False
    i = 0
    with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
        for record in ijson.items(f, "results.item"):
            try:
                i += 1
                if record['primaryAccession'] == uniprot_acc:
                    print("found")
                    sequence = record['sequence']['value']
                    refs = record.get("uniProtKBCrossReferences", [])
                    go = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                    found = True
                    break

                if i % 10000 == 0:
                    print(i)
                    
            except Exception as record_error:
                print("Error processing record:", record_error)
    
    if found == False:
        go = []
    return find_e3_from_sequence(sequence, number_outputs, found, go)

###################################################################

## Kinase Interaction Finder ##

dic_kinase = {}
with open('kinase/dic_kinase.json') as f:
    dic_kinase = json.load(f)

dic_go_kinase = {}
with open('kinase/dic_GO_problem_3.json') as f:
    dic_go_kinase = json.load(f)

inv_dic_kinase = {v: k for k, v in dic_kinase.items()}

class model_embedding_pep(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(model_embedding_pep, self).__init__()
        self.fc_embed = nn.Linear(embed_dim, 2048)
        self.embed = nn.Embedding(26, 128)
        self.gru = nn.GRU(128, 256, 2, batch_first=True)
        self.fc_pep = nn.Linear(256, 2048)
        self.fc1 = nn.Linear(2048 * 2, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def forward(self, embed, pep, site):
        x_embed = F.relu(self.fc_embed(embed))
        x_pep = self.embed(pep)
        x_pep, _ = self.gru(x_pep)
        x_pep = x_pep[:,-1,:]
        x_pep = F.relu(self.fc_pep(x_pep))
        x = torch.cat((x_embed, x_pep), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    

class model_embedding_pep_go(nn.Module):
    def __init__(self, embed_dim, go_dim, output_dim):
        super(model_embedding_pep_go, self).__init__()
        self.fc_embed = nn.Linear(embed_dim, 2048)
        self.embed = nn.Embedding(26, 128)
        self.gru = nn.GRU(128, 256, 2, batch_first=True)
        self.fc_pep = nn.Linear(256, 2048)
        self.fc_go = nn.Linear(go_dim, 2048)
        self.fc1 = nn.Linear(2048 * 3, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def forward(self, embed, pep, site, go):
        x_embed = F.relu(self.fc_embed(embed))
        x_pep = self.embed(pep)
        x_pep, _ = self.gru(x_pep)
        x_pep = x_pep[:,-1,:]
        x_pep = F.relu(self.fc_pep(x_pep))
        x_go = F.relu(self.fc_go(go))
        x = torch.cat((x_embed, x_pep, x_go), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

model_kinase_without_go = model_embedding_pep(1024, 455)
model_kinase_with_go = model_embedding_pep_go(1024, 2000, 455)

model_kinase_without_go.load_state_dict(torch.load('kinase/model_embedding_pep_site_problem_3_final.pt'))
model_kinase_with_go.load_state_dict(torch.load('kinase/model_embedding_pep_go_site_problem_3_final.pt'))

# Function to find Kinase interactions from the model
def find_kinase_from_sequence(sequence, cleavage_site, number_outputs = 3, swissprot = '?', go = []):
    sequence_examples = [sequence]
    sequence_Example = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    ids = tokenizer.batch_encode_plus(
        sequence_Example, add_special_tokens=True, padding="longest"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        model_prott5.to(device)

        with torch.no_grad():
            embeddings = model_prott5(
                input_ids=tokenized_sequences, attention_mask=attention_mask
            )

        # Extract embeddings for the first sequence in the batch while removing padded & special tokens
        emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

        # Derive a single representation (per-protein embedding) for the whole protein
        emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

        embedding = emb_0_per_protein  # shape (1024)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory, switching to CPU")
            device = 'cpu'
            tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            model_prott5.to(device)

            with torch.no_grad():
                embeddings = model_prott5(
                    input_ids=tokenized_sequences, attention_mask=attention_mask
                )

            # Extract embeddings for the first sequence in the batch while removing padded & special tokens
            emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

            # Derive a single representation (per-protein embedding) for the whole protein
            emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

            embedding = emb_0_per_protein  # shape (1024)
        else:
            raise e
    seq = sequence
    p1 = cleavage_site
    pep = []
    min = p1 - 10
    max = p1 + 10

    for j in range(min, max):
        if j < 0 or j >= len(seq):
            pep.append('X')
        else:
            pep.append(seq[j])

    vocab = ['A', 'B', 'C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}

    pep_int = [vocab_dict[pep[j]] for j in range(len(pep))]

    embed = embedding.unsqueeze(0)
    cleav_embedding = torch.tensor(pep_int, dtype=torch.long)
    cleavage_site = torch.tensor(cleavage_site, dtype=torch.float32)

    embed = embed.to(device)
    cleav_embedding = cleav_embedding.unsqueeze(0).to(device)
    cleavage_site = cleavage_site.unsqueeze(0).unsqueeze(1).to(device)

    if swissprot == True:
        model_kinase_with_go.to(device)
        go_int = [0 for i in range(2000)]
        for i in range(len(go)):
            if go[i] in dic_go_kinase.keys():
                go_int[dic_go_kinase[go[i]]] = 1
        go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
        
        outputs = model_kinase_with_go(embed, cleav_embedding, cleavage_site, go_tensor)

    if swissprot == False:
        model_kinase_without_go.to(device)
        outputs = model_kinase_without_go(embed, cleav_embedding, cleavage_site)
    
    if swissprot == '?':
        go = go_from_seq(sequence)
        if go == []:
            model_kinase_without_go.to(device)
            outputs = model_kinase_without_go(embed,cleav_embedding, cleavage_site)
        else:
            model_kinase_with_go.to(device)
            go_int = [0 for i in range(2000)]
            for i in range(len(go)):
                if go[i] in dic_go_kinase.keys():
                    go_int[dic_go_kinase[go[i]]] = 1
            go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model_kinase_with_go(embed, cleav_embedding, cleavage_site, go_tensor)  

    top_indices = np.argsort(outputs[0].cpu().detach().numpy())[-number_outputs:][::-1]
    results = []

    for idx in top_indices:
        category = inv_dic_kinase[idx]
        probability = outputs[0][idx]
        results.append({"Category": category, "Probability": round(probability.item(), 4)})

    return results

def find_kinase(uniprot_acc, cleavage_site, number_outputs=3):
    found = False
    i = 0

    with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
        for record in ijson.items(f, "results.item"):
            try:
                i += 1
                if record['primaryAccession'] == uniprot_acc:
                    print("found")
                    sequence = record['sequence']['value']
                    refs = record.get("uniProtKBCrossReferences", [])
                    go = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                    found = True
                    break

                if i % 10000 == 0:
                    print(i)
                    
            except Exception as record_error:
                print("Error processing record:", record_error)
    
    if found == False:
        go = []
    return find_kinase_from_sequence(sequence, cleavage_site, number_outputs, found, go)

###################################################################

## Protease Finder ##

dic_enzyme = {}
with open('protease/dic_enzyme.json') as f:
    dic_enzyme = json.load(f)

dic_go = {}
with open('protease/dic_GO_problem_1.json') as f:
    dic_go = json.load(f)

inv_dic_enzyme = {v: k for k, v in dic_enzyme.items()}

class model_embedding_pep_site(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(model_embedding_pep_site, self).__init__()
        self.fc_embed = nn.Linear(embed_dim, 2048)
        self.embed = nn.Embedding(26, 128)
        self.gru = nn.GRU(128, 256, 2, batch_first=True)
        self.fc_pep = nn.Linear(256, 2048)
        self.fc_site = nn.Linear(1, 10)
        self.fc1 = nn.Linear(2048 * 2 + 10, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def forward(self, embed, pep, site):
        x_embed = F.relu(self.fc_embed(embed))
        x_pep = self.embed(pep)
        x_pep, _ = self.gru(x_pep)
        x_pep = x_pep[:,-1,:]
        x_pep = F.relu(self.fc_pep(x_pep))
        x_site = F.relu(self.fc_site(site))
        x = torch.cat((x_embed, x_pep, x_site), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class model_embedding_pep_go_site(nn.Module):
    def __init__(self, embed_dim, go_dim, output_dim):
        super(model_embedding_pep_go_site, self).__init__()
        self.fc_embed = nn.Linear(embed_dim, 2048)
        self.embed = nn.Embedding(26, 128)
        self.gru = nn.GRU(128, 256, 2, batch_first=True)
        self.fc_pep = nn.Linear(256, 2048)
        self.fc_go = nn.Linear(go_dim, 2048)
        self.fc_site = nn.Linear(1, 10)
        self.fc1 = nn.Linear(2048 * 3 + 10, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def forward(self, embed, pep, site, go):
        x_embed = F.relu(self.fc_embed(embed))
        x_pep = self.embed(pep)
        x_pep, _ = self.gru(x_pep)
        x_pep = x_pep[:,-1,:]
        x_pep = F.relu(self.fc_pep(x_pep))
        x_go = F.relu(self.fc_go(go))
        x_site = F.relu(self.fc_site(site))
        x = torch.cat((x_embed, x_pep, x_go, x_site), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

model_protease_without_go = model_embedding_pep_site(1024, 535)
model_protease_with_go = model_embedding_pep_go_site(1024, 2000, 535)

model_protease_without_go.load_state_dict(torch.load('protease/model_embedding_pep_site_problem_1_final.pt'))
model_protease_with_go.load_state_dict(torch.load('protease/model_embedding_pep_go_site_problem_1_final.pt'))

# Charger les modèles T5 pour les embeddings de séquences
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
model_prott5 = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')

def find_protease_from_sequence(sequence, cleavage_site, number_outputs=3, swissprot='?', go = []):
    sequence_examples = [sequence]
    sequence_Example = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    ids = tokenizer.batch_encode_plus(
        sequence_Example, add_special_tokens=True, padding="longest"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        model_prott5.to(device)

        with torch.no_grad():
            embeddings = model_prott5(
                input_ids=tokenized_sequences, attention_mask=attention_mask
            )

        # Extract embeddings for the first sequence in the batch while removing padded & special tokens
        emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

        # Derive a single representation (per-protein embedding) for the whole protein
        emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

        embedding = emb_0_per_protein  # shape (1024)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory, switching to CPU")
            device = 'cpu'
            tokenized_sequences = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            model_prott5.to(device)

            with torch.no_grad():
                embeddings = model_prott5(
                    input_ids=tokenized_sequences, attention_mask=attention_mask
                )

            # Extract embeddings for the first sequence in the batch while removing padded & special tokens
            emb_0 = embeddings[0][0, :len(sequence_examples[0])]  # shape (7 x 1024)

            # Derive a single representation (per-protein embedding) for the whole protein
            emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

            embedding = emb_0_per_protein  # shape (1024)
        else:
            raise e
    seq = sequence
    p1 = cleavage_site
    pep = []
    min = p1 - 10
    max = p1 + 10

    for j in range(min, max):
        if j < 0 or j >= len(seq):
            pep.append('X')
        else:
            pep.append(seq[j])

    vocab = ['A', 'B', 'C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}

    pep_int = [vocab_dict[pep[j]] for j in range(len(pep))]

    embed = embedding.unsqueeze(0)
    cleav_embedding = torch.tensor(pep_int, dtype=torch.long)
    cleavage_site = torch.tensor(cleavage_site, dtype=torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embed = embed.to(device)
    cleav_embedding = cleav_embedding.unsqueeze(0).to(device)
    cleavage_site = cleavage_site.unsqueeze(0).unsqueeze(1).to(device)

    if swissprot == True:
        model_protease_with_go.to(device)
        go_int = [0 for i in range(2000)]
        for i in range(len(go)):
            if go[i] in dic_go.keys():
                go_int[dic_go[go[i]]] = 1
        go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
        
        outputs = model_protease_with_go(embed, cleav_embedding, cleavage_site, go_tensor)

    if swissprot == False:
        model_protease_without_go.to(device)
        outputs = model_protease_without_go(embed, cleav_embedding, cleavage_site)
    
    if swissprot == '?':
        go = go_from_seq(sequence)
        if go == []:
            model_protease_without_go.to(device)
            outputs = model_protease_without_go(embed, cleav_embedding, cleavage_site)
        else:
            model_protease_with_go.to(device)
            go_int = [0 for i in range(2000)]
            for i in range(len(go)):
                if go[i] in dic_go.keys():
                    go_int[dic_go[go[i]]] = 1
            go_tensor = torch.tensor(go_int, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model_protease_with_go(embed, cleav_embedding, cleavage_site, go_tensor)  

    top_indices = np.argsort(outputs[0].cpu().detach().numpy())[-number_outputs:][::-1]
    results = []

    for idx in top_indices:
        category = inv_dic_enzyme[idx]
        probability = outputs[0][idx]
        results.append({"Category": category, "Probability": round(probability.item(), 4)})
    return results

def go_from_seq(sequence):
    found = False

    with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
        for record in ijson.items(f, "results.item"):
            try:
                if record['sequence']['value'] == sequence:
                    refs = record.get("uniProtKBCrossReferences", [])
                    go = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                    found = True
                    break
                    
            except Exception as record_error:
                print("Error processing record:", record_error)
    
    if found == False:
        go = []
    return go

def find_protease(uniprot_acc, cleavage_site, number_outputs=3):
    found = False
    i = 0

    with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
        for record in ijson.items(f, "results.item"):
            try:
                i += 1
                if record['primaryAccession'] == uniprot_acc:
                    print("found")
                    sequence = record['sequence']['value']
                    refs = record.get("uniProtKBCrossReferences", [])
                    go = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                    found = True
                    break

                if i % 10000 == 0:
                    print(i)
                    
            except Exception as record_error:
                print("Error processing record:", record_error)
    
    if found == False:
        go = []
    return find_protease_from_sequence(sequence, cleavage_site, number_outputs, found, go)

###################################################################

# Interface Streamlit

st.title('Interaction Finder')

def plot_results(results, ):
    categories = [result['Category'] for result in results]
    probabilities = [result['Probability'] for result in results]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=categories, y=probabilities)
    plt.xlabel('Category')
    plt.ylabel('Probability')
    plt.title('Predicted Enzymes and Their Probabilities')
    for index, value in enumerate(probabilities):
        plt.text(index, value, f'{value:.2f}', ha='center')
    st.pyplot(plt)

# Selector for interaction type
interaction_type = st.radio("Choose interaction type", ["Kinase", "E3 Ligase", "Protease"])

#Proteases
if interaction_type == 'Protease':
    # Option to choose mode
    mode = st.radio("Choose the mode of input", ["Single Entry", "Batch Upload"])

    if mode == "Single Entry":
        # Selector for input mode
        input_mode = st.selectbox("Choose input mode", ["UniProt Accession", "Sequence"])

        # Inputs for protein based on selected mode
        if input_mode == "UniProt Accession":
            protein = st.text_input('Enter the UniProt Accession')
        else:
            sequence = st.text_area('Enter the protein sequence')

        # Input for cleavage site
        site = st.number_input('Enter the cleavage site', min_value=1, step=1)

        # Number of outputs
        number_outputs = st.number_input('Number of predicted proteases', min_value=1, step=1)

        # Button to submit the information
        if st.button('Search'):
            if input_mode == "UniProt Accession" and protein and site:
                results = find_protease(protein, site, number_outputs)
            elif input_mode == "Sequence" and sequence and site:
                results = find_protease_from_sequence(sequence, site, number_outputs)
            else:
                results = "Please enter both the cleavage site and the protein."

            if isinstance(results, str):
                st.write(results)
            else:
                st.write("Results found:")
                plot_results(results)

    elif mode == "Batch Upload":
        st.write("Please make sure your Excel file has a column for UniProt Accession or Sequence and p1 Position.")
        # File uploader for batch predictions
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        
        # Number of outputs
        number_outputs = st.number_input('Number of predicted proteases', min_value=1, step=1)

        # Handling batch predictions from an Excel file
        if uploaded_file and st.button('Process File'):
            df = pd.read_excel(uploaded_file)
            if 'UniProt Accession' in df.columns or 'Sequence' in df.columns and 'p1 Position' in df.columns:
                st.write("Processing the uploaded file...")
                
                # Create a dictionary to hold the protein information
                list_protein = []
                dic_protein_test = {}

                # Initialize dictionary entries for UniProt Accessions in the input file
                if 'UniProt Accession' in df.columns:
                    for uniprot_acc in df['UniProt Accession']:
                        if not pd.isna(uniprot_acc):
                            list_protein.append(uniprot_acc.split(';')[0].strip())

                # Process the UniProtKB JSON file once to gather GO terms and sequences
                i = 0
                with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
                    for record in ijson.items(f, "results.item"):
                        try:
                            i += 1
                            refs = record.get("uniProtKBCrossReferences", [])
                            if record["primaryAccession"] in list_protein:
                                GO = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                                sequence = record["sequence"]["value"]
                                dic_protein_test[record["primaryAccession"]] = [GO, sequence]
                            
                            if i % 10000 == 0:
                                print(i)
                        
                        except Exception as record_error:
                            print("Error processing record:", record_error)

                # Process each row in the input file and gather results
                all_results = []
                i = 0
                for _, row in df.iterrows():
                    i +=1
                    if i%10 == 0:
                        print(i)
                    if pd.isna(row['p1 Position']):
                        site = 0
                    elif type(row['p1 Position']) == str:
                        site = int(row['p1 Position'].split(';')[0])
                    else:
                        site = int(row['p1 Position'])
                    if 'UniProt Accession' in df.columns and not pd.isna(row['UniProt Accession']):
                        uniprot_acc = row['UniProt Accession'].split(';')[0].strip()
                        if uniprot_acc in dic_protein_test:
                            go, sequence = dic_protein_test[uniprot_acc]
                            results = find_protease_from_sequence(sequence, site, number_outputs, swissprot=True, go=go)
                        else:
                            # Si UniProt Accession n'est pas trouvé, vérifier s'il y a une colonne Sequence
                            if 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                                sequence = row['Sequence']
                                results = find_protease_from_sequence(sequence, site, number_outputs, swissprot=False)
                            else:
                                results = "UniProt Accession not found in the UniProtKB JSON and no Sequence column available."
                    elif 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                        sequence = row['Sequence']
                        results = find_protease_from_sequence(sequence, site, number_outputs, swissprot=False)
                    else:
                        results = "Please make sure your Excel file has the correct columns for the selected input mode."

                    if isinstance(results, str):
                        all_results.append([results])
                    else:
                        result_row = []
                        for result in results:
                            result_row.append(result['Category'])
                            result_row.append(result['Probability'])
                        # Ensure each row has enough columns
                        all_results.append(result_row)

                columns = []
                for i in range(number_outputs):
                    columns += [f'Category {i+1}'] + [f'Probability {i+1}']
                results_df = pd.DataFrame(all_results, columns=columns)
                #Concatenate the results
                results_df = pd.concat([df, results_df], axis=1)
                
                # Save DataFrame to Excel
                output_file = "results.xlsx"
                results_df.to_excel(output_file, index=False)
                
                # Provide a download link
                st.write("Batch processing complete. Download the results:")
                st.download_button(
                    label="Download Excel file",
                    data=open(output_file, "rb").read(),
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.write("Please make sure your Excel file has 'UniProt Accession' or 'Sequence' and 'p1 Position' columns.")

# E3 Ligases
elif interaction_type == 'E3 Ligase':
    mode = st.radio("Choose the mode of input", ["Single Entry", "Batch Upload"])
    if mode == "Single Entry":
        # Input for UniProt Accession
        uniprot_acc = st.text_input('Enter the UniProt Accession')
        
        # Number of outputs
        number_outputs = st.number_input('Number of predicted E3 interactions', min_value=1, step=1)

        # Button to submit the information
        if st.button('Find E3 Interactions'):
            if uniprot_acc:
                results = find_e3(uniprot_acc, number_outputs=number_outputs)
                st.write("Results found:")
                plot_results(results)
            else:
                st.write("Please enter the UniProt Accession.")

    elif mode == "Batch Upload":
        st.write("Please make sure your Excel file has a column for UniProt Accession or Sequence.")
        # File uploader for batch predictions
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        
        # Number of outputs
        number_outputs = st.number_input('Number of predicted E3 interactions', min_value=1, step=1)

        # Handling batch predictions from an Excel file
        if uploaded_file and st.button('Process File'):
            df = pd.read_excel(uploaded_file)
            if 'UniProt Accession' in df.columns or 'Sequence' in df.columns:
                st.write("Processing the uploaded file...")
                
                # Create a dictionary to hold the protein information
                dic_protein_test = {}
                list_protein = []

                # Initialize dictionary entries for UniProt Accessions in the input file
                if 'UniProt Accession' in df.columns:
                    for uniprot_acc in df['UniProt Accession']:
                        if not pd.isna(uniprot_acc):
                            list_protein.append(uniprot_acc.split(';')[0].strip())

                # Process the UniProtKB JSON file once to gather GO terms and sequences
                i = 0
                with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
                    for record in ijson.items(f, "results.item"):
                        try:
                            i += 1
                            refs = record.get("uniProtKBCrossReferences", [])
                            if record["primaryAccession"] in list_protein:
                                GO = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                                sequence = record["sequence"]["value"]
                                dic_protein_test[record["primaryAccession"]] = [GO, sequence]
                            
                            if i % 10000 == 0:
                                print(i)
                        
                        except Exception as record_error:
                            print("Error processing record:", record_error)

                # Process each row in the input file and gather results
                all_results = []
                for _, row in df.iterrows():
                    if 'UniProt Accession' in df.columns and not pd.isna(row['UniProt Accession']):
                        uniprot_acc = row['UniProt Accession'].split(';')[0].strip()
                        if uniprot_acc in dic_protein_test:
                            go, sequence = dic_protein_test[uniprot_acc]
                            results = find_e3_from_sequence(sequence, number_outputs, swissprot=True, go=go)
                        else:
                            # Si UniProt Accession n'est pas trouvé, vérifier s'il y a une colonne Sequence
                            if 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                                sequence = row['Sequence']
                                results = find_e3_from_sequence(sequence, number_outputs, swissprot=False)
                            else:
                                results = "UniProt Accession not found in the UniProtKB JSON and no Sequence column available."
                    elif 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                        sequence = row['Sequence']
                        results = find_e3_from_sequence(sequence, number_outputs, swissprot=False)
                    else:
                        results = "Please make sure your Excel file has the correct columns for the selected input mode."

                    if isinstance(results, str):
                        all_results.append([results])
                    else:
                        result_row = []
                        for result in results:
                            result_row.append(result['Category'])
                            result_row.append(result['Probability'])
                        # Ensure each row has enough columns
                        all_results.append(result_row)

                columns = []
                for i in range(number_outputs):
                    columns += [f'Category {i+1}'] + [f'Probability {i+1}']
                results_df = pd.DataFrame(all_results, columns=columns)
                results_df = pd.concat([df, results_df], axis=1)
                # Save DataFrame to Excel
                output_file = "results.xlsx"
                results_df.to_excel(output_file, index=False)
                
                # Provide a download link
                st.write("Batch processing complete. Download the results:")
                st.download_button(
                    label="Download Excel file",
                    data=open(output_file, "rb").read(),
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.write("Please make sure your Excel file has 'UniProt Accession' or 'Sequence' column.")

# Kinases
elif interaction_type == 'Kinase':
    # Option to choose mode
    mode = st.radio("Choose the mode of input", ["Single Entry", "Batch Upload"])

    if mode == "Single Entry":
        # Input for UniProt Accession and site
        uniprot_acc = st.text_input('Enter the UniProt Accession')
        site = st.number_input('Enter the site', min_value=1, step=1)

        # Number of outputs
        number_outputs = st.number_input('Number of predicted kinase interactions', min_value=1, step=1)

        # Button to submit the information
        if st.button('Find Kinase Interactions'):
            if uniprot_acc and site:
                results = find_kinase(uniprot_acc, site, number_outputs=number_outputs)
                st.write("Results found:")
                plot_results(results)
            else:
                st.write("Please enter the UniProt Accession and site.")

    elif mode == "Batch Upload":
        # Write that the uploaded file has to have a column for UniProt Accession or Sequence and p1 Position
        st.write("Please make sure your Excel file has a column for 'UniProt Accession' or 'Sequence' and 'p1 Position'.")
        # File uploader for batch predictions
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        
        # Number of outputs
        number_outputs = st.number_input('Number of predicted kinase interactions', min_value=1, step=1)

        # Handling batch predictions from an Excel file
        if uploaded_file and st.button('Process File'):
            df = pd.read_excel(uploaded_file)
            if 'UniProt Accession' in df.columns or 'Sequence' in df.columns and 'p1 Position' in df.columns:
                st.write("Processing the uploaded file...")
                
                # Create a dictionary to hold the protein information
                dic_protein_test = {}
                list_protein = []
                # Initialize dictionary entries for UniProt Accessions in the input file
                if 'UniProt Accession' in df.columns:
                    for uniprot_acc in df['UniProt Accession']:
                        if not pd.isna(uniprot_acc):
                            list_protein.append(uniprot_acc.split(';')[0].strip())

                # Process the UniProtKB JSON file once to gather GO terms and sequences
                i = 0
                with open('uniprotkb_AND_reviewed_true_2024_03_26.json', "rb") as f:
                    for record in ijson.items(f, "results.item"):
                        try:
                            i += 1
                            refs = record.get("uniProtKBCrossReferences", [])
                            if record["primaryAccession"] in list_protein:
                                GO = [ref["id"] for ref in refs if ref.get("database") == "GO"]
                                sequence = record["sequence"]["value"]
                                dic_protein_test[record["primaryAccession"]] = [GO, sequence]
                            
                            if i % 10000 == 0:
                                print(i)
                        
                        except Exception as record_error:
                            print("Error processing record:", record_error)
                # Process each row in the input file and gather results
                all_results = []
                i = 0
                for _, row in df.iterrows():
                    i += 1
                    if i % 10 == 0:
                        print(i)
                    if pd.isna(row['p1 Position']):
                        site = 0
                    elif type(row['p1 Position']) == str:
                        site = int(row['p1 Position'].split(';')[0])
                    else:
                        site = int(row['p1 Position'])
                    if 'UniProt Accession' in df.columns and not pd.isna(row['UniProt Accession']):
                        uniprot_acc = row['UniProt Accession'].split(';')[0].strip()
                        if uniprot_acc in dic_protein_test:
                            go, sequence = dic_protein_test[uniprot_acc]
                            results = find_kinase_from_sequence(sequence, site, number_outputs, swissprot=True, go=go)
                        else:
                            # Si UniProt Accession n'est pas trouvé, vérifier s'il y a une colonne Sequence
                            if 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                                sequence = row['Sequence']
                                results = find_kinase_from_sequence(sequence, site, number_outputs, swissprot=False)
                            else:
                                results = "UniProt Accession not found in the UniProtKB JSON and no Sequence column available."
                    elif 'Sequence' in df.columns and not pd.isna(row['Sequence']):
                        sequence = row['Sequence']
                        results = find_kinase_from_sequence(sequence, site, number_outputs, swissprot=False)
                    else:
                        results = "Please make sure your Excel file has the correct columns for the selected input mode."
                    if isinstance(results, str):
                        all_results.append([results])
                    else:
                        result_row = []
                        for result in results:
                            result_row.append(result['Category'])
                            result_row.append(result['Probability'])
                        # Ensure each row has enough columns
                        all_results.append(result_row)

                columns = []
                for i in range(number_outputs):
                    columns += [f'Category {i+1}'] + [f'Probability {i+1}']
                results_df = pd.DataFrame(all_results, columns=columns)
                results_df = pd.concat([df, results_df], axis=1)

                # Save DataFrame to Excel
                output_file = "results.xlsx"
                results_df.to_excel(output_file, index=False)
                
                # Provide a download link
                st.write("Batch processing complete. Download the results:")
                st.download_button(
                    label="Download Excel file",
                    data=open(output_file, "rb").read(),
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.write("Please make sure your Excel file has 'UniProt Accession' or 'Sequence' and 'p1 Position' columns.")
