# validate_dataset.py

from datasets import load_dataset

# Carrega o dataset
dataset = load_dataset("isaiahbjork/chain-of-thought")

# Inspeciona as primeiras 3 entradas
for i in range(3):
    print(f"Entrada {i+1}:")
    print(dataset['train'][i])
    print("\n" + "-"*50 + "\n")