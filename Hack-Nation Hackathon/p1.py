import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_idx = {a:i for i,a in enumerate(amino_acids)}
label_map = {'H':0,'E':1,'C':2}

def encode_sequence(seq, max_len):
    arr = np.zeros((max_len, 20), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        arr[i, aa_to_idx[aa]] = 1.0
    return arr

def encode_labels(labels, max_len):
    arr = np.full(max_len, 2, dtype=np.int64)
    for i, l in enumerate(labels[:max_len]):
        arr[i] = label_map[l]
    return arr

class ProteinDataset(Dataset):
    def __init__(self, seq_file, label_file, max_len=128):
        self.seqs = open(seq_file).read().splitlines()
        self.labels = open(label_file).read().splitlines()
        self.max_len = max_len
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        labels = self.labels[idx]
        x = encode_sequence(seq, self.max_len)
        y = encode_labels(labels, self.max_len)
        return torch.tensor(x), torch.tensor(y)

dataset = ProteinDataset("seq.txt", "label.txt", max_len=128)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

class CNNSecondaryStructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(20, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 3)
    def forward(self, x):
        x = x.permute(0,2,1)  
        x = self.relu(self.conv(x))  
        x = x.permute(0,2,1)        
        logits = self.fc(x)          
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSecondaryStructure().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.view(-1, 3)
        labels = y.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(y.numpy().flatten())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(all_labels, all_preds)
print("Validation Accuracy:", acc)
print(classification_report(all_labels, all_preds, target_names=['Helix','Sheet','Coil']))

import matplotlib.pyplot as plt
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(['Helix','Sheet','Coil'])
ax.set_yticklabels(['Helix','Sheet','Coil'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




import py3Dmol

view = py3Dmol.view(query='pdb:1CRN')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.setBackgroundColor('white')
view.zoomTo()
view.spin(True) 
view.show()
with open('your_protein.pdb', 'r') as f:
    pdb_data = f.read()

view = py3Dmol.view()
view.addModel(pdb_data, 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.setBackgroundColor('white')
view.zoomTo()
view.spin(True)
view.show()

