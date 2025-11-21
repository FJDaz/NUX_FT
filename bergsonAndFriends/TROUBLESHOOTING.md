# 🔧 Troubleshooting - Problèmes Courants

## ❌ Problème: Cellule 1 tourne dans le vide (pas d'output)

### Symptôme
La cellule d'installation (`!pip install -q -U ...`) tourne indéfiniment sans aucun feedback :
- Pas d'erreur
- Pas de progression visible
- On dirait que ça freeze

### Cause
Le flag `-q` (quiet mode) masque tout l'output de pip, donnant l'impression que rien ne se passe. L'installation prend réellement **2-3 minutes** pour télécharger et installer les packages.

### ✅ Solution
**Le notebook a été corrigé** : flag `-q` retiré pour afficher la progression.

**Maintenant vous verrez :**
```
📦 Installation des packages (peut prendre 2-3 minutes)...

Collecting torch>=2.2.0
  Downloading torch-2.5.1-cp310-cp310-linux_x86_64.whl (1024 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1024/1024 MB 15.2 MB/s eta 0:00:00
Installing collected packages: torch
...
✅ Installation terminée !
```

**Temps normal :**
- GPU T4/V100/A100 : 2-3 minutes
- Première exécution : plus long (téléchargements)
- Exécutions suivantes : plus rapide (cache)

---

## ❌ Erreur: "No matching distribution found for torch==2.1.2"

### Problème
```
ERROR: Could not find a version that satisfies the requirement torch==2.1.2
ERROR: No matching distribution found for torch==2.1.2
```

### Cause
PyTorch 2.1.2 n'existe plus dans les dépôts pip (version obsolète).

### ✅ Solution
**Le notebook a été corrigé** avec des versions récentes :
- `torch>=2.2.0` (au lieu de `torch==2.1.2`)
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `bitsandbytes>=0.43.0`

**Si vous utilisez une vieille version du notebook :**
1. Re-télécharger le notebook depuis ce repo
2. Ou modifier manuellement la cellule 2 :
   ```python
   !pip install -q -U torch>=2.2.0 transformers>=4.40.0 peft>=0.10.0 ...
   ```

---

## ❌ Erreur: "CUDA out of memory" sur Colab

### Problème
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

### Solutions

#### Option 1 : Réduire batch size
Dans la cellule après vérification GPU, forcer un batch size plus petit :
```python
BATCH_SIZE = 2  # Au lieu de 4 ou 8
GRADIENT_ACCUM = 16  # Compenser (effective batch = 32)
```

#### Option 2 : Redémarrer avec GPU plus puissant
1. Runtime > Disconnect and delete runtime
2. Runtime > Change runtime type > **A100 GPU** (ou V100)
3. Relancer "Run all"

#### Option 3 : Réduire dataset
Utiliser seulement `schemes_levelA_base.jsonl` (300 exemples au lieu de 1200) :
```python
# Commenter la ligne d'augmentation
# dataset_augmented = load_dataset(...)
dataset_full = dataset_base  # Seulement 300 exemples
```

---

## ❌ Erreur: "SFTTrainer not found"

### Problème
```
ImportError: cannot import name 'SFTTrainer' from 'trl'
```

### Cause
Version de `trl` trop ancienne.

### Solution
```python
!pip install -U trl>=0.8.0
```

---

## ❌ Training très lent (>3h sur T4)

### Solutions

#### Option 1 : Réduire epochs
```python
num_train_epochs=2  # Au lieu de 3
```

#### Option 2 : Réduire dataset
Utiliser seulement 300 exemples base (au lieu de 1200).

#### Option 3 : Utiliser GPU plus rapide
Colab Pro permet d'accéder à A100 (10x plus rapide que T4).

---

## ❌ Eval loss > 1.0 (ne converge pas)

### Problème
Après training, la validation loss reste élevée (>1.0).

### Causes possibles
1. **Learning rate trop élevé** → Réduire à `1e-4` (au lieu de `2e-4`)
2. **Pas assez d'epochs** → Augmenter à 4 ou 5
3. **Dataset corrompu** → Vérifier format JSONL

### Solution
Relancer training avec learning rate réduit :
```python
learning_rate=1e-4,  # Plus conservateur
num_train_epochs=4,  # Plus d'epochs
```

---

## ❌ LoRA ne charge pas correctement en local

### Problème
```python
PeftModel.from_pretrained(model, lora_path)
# Erreur ou pas d'amélioration vs base model
```

### Solutions

#### Vérifier fichiers LoRA
```bash
ls -lh models/mistral-7b-philosophes-lora-final/
# Doit contenir:
# - adapter_config.json
# - adapter_model.safetensors
```

#### Forcer merge
```python
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # Merge explicite
```

#### Tester avec inférence simple
```python
# Test rapide
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe("Schème : Modus Ponens\nContexte : Si P alors Q, or P\nApplique :")
print(result)
```

---

## ❌ Latence CPU > 30s par réponse

### Problème
Inférence trop lente sur CPU (>30s par réponse).

### Solutions

#### Option 1 : Vérifier quantization 4-bit
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Doit être True
    # ...
)
```

#### Option 2 : Réduire max_new_tokens
```python
max_new_tokens=64,  # Au lieu de 128
```

#### Option 3 : Utiliser GPU
Si latence critique, déployer sur Modal/HF GPU au lieu de CPU.

---

## ❌ Push vers HF Hub échoue

### Problème
```
403 Forbidden: Token does not have write access
```

### Solution
1. Aller sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Créer un **nouveau token** avec **write access** ✅
3. Remplacer dans le notebook :
   ```python
   HF_TOKEN = "hf_NOUVEAU_TOKEN_AVEC_WRITE_ACCESS"
   ```

---

## ❌ Modèle génère seulement des zéros après re-fine-tuning

### Problème
Après re-fine-tuning, le modèle génère des outputs corrompus (répétition de zéros) :
```
💬 SPINOZA : [20000000000000000000000000000000000000000...
```

### Cause
**Catastrophic forgetting massif** causé par :
- Re-fine-tuning sur dataset trop petit (ex: 23 exemples)
- Ratio déséquilibré : 1200 exemples initiaux → 23 exemples correction
- Le modèle "oublie" tout ce qu'il a appris et surajuste sur les 23 exemples

### ✅ Solution
**TOUJOURS combiner datasets** lors du re-fine-tuning :

1. **Ratio 80/20 :**
   - 80% du dataset original (schèmes logiques)
   - 100% du dataset de correction (incarnation)
   - Exemple : 720 schèmes + 213 incarnation = 933 exemples

2. **Code correct :**
   ```python
   from datasets import concatenate_datasets

   # Prendre 80% du dataset original
   dataset_schemes_sample = dataset_schemes.shuffle(seed=42).select(range(int(len(dataset_schemes)*0.8)))

   # Combiner avec dataset correction
   dataset_combined = concatenate_datasets([dataset_schemes_sample, dataset_incarnation])
   ```

3. **Paramètres adaptés :**
   - Learning rate normal : `2e-4` (pas réduit)
   - Epochs : 2-3 (pas 1 seul)
   - Monitoring : `eval_loss` tous les 20 steps

**Voir Section 8 du notebook** pour l'implémentation complète.

---

## 📞 Aide Supplémentaire

Si aucune solution ne fonctionne :

1. **Vérifier versions :**
   ```python
   import torch, transformers, peft
   print(f"torch: {torch.__version__}")
   print(f"transformers: {transformers.__version__}")
   print(f"peft: {peft.__version__}")
   ```

2. **Consulter documentation :**
   - [USAGE.md](USAGE.md) - Guide complet
   - [QUICKSTART.md](QUICKSTART.md) - Guide rapide
   - [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Synthèse projet

3. **Logs détaillés :**
   Dans le notebook, activer logging verbeux :
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

---

**Dernière mise à jour :** 20 novembre 2025
**Versions testées :** torch==2.8.0, transformers>=4.40.0, peft>=0.10.0, trl>=0.8.0
