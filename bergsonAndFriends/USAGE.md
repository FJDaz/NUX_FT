# 📖 Guide d'Utilisation - Fine-Tuning Mistral 7B

## 🚀 Quick Start

### 1. Training sur Colab Pro

#### Étape 1 : Uploader les fichiers
1. Ouvrir [Google Colab](https://colab.research.google.com/)
2. Upload le notebook : `notebooks/train_mistral_7b_lora.ipynb`
3. Sélectionner GPU : **Runtime > Change runtime type > A100 GPU** (ou V100/T4)
4. Upload les datasets dans Colab :
   - `data/FT/processed/schemes_levelA_base.jsonl`
   - `data/FT/processed/schemes_levelA_augmented.jsonl`

#### Étape 2 : Configuration
1. Dans la cellule "Configuration", remplacer :
   ```python
   HF_TOKEN = "hf_..."  # Votre token HF (avec write access)
   HF_USERNAME = "FJDaz"  # Votre username HF
   ```

#### Étape 3 : Lancer le training
1. Cliquer sur **Runtime > Run all**
2. Attendre la fin du training :
   - **A100 40GB :** 30-45 minutes ⚡
   - **V100 16GB :** 1h-1h30
   - **T4 15GB :** 2-3h

#### Étape 4 : Récupérer le modèle
**Option A : Download local**
```python
# Dernière cellule du notebook
!zip -r mistral-7b-philosophes-lora-final.zip ./mistral-7b-philosophes-lora-final/
# Télécharger depuis l'explorateur Colab (gauche)
```

**Option B : Push vers HF Hub** (automatique si configuré)
- Votre LoRA sera disponible sur : `https://huggingface.co/FJDaz/mistral-7b-philosophes-lora`

---

## 🧪 Test Local du Modèle

### Prérequis
```bash
cd /Users/francois-jeandazin/NUX_FT/bergsonAndFriends

# Installer les dépendances
pip install torch transformers peft bitsandbytes accelerate
```

### Tester le LoRA

```bash
# Copier le LoRA téléchargé depuis Colab
unzip mistral-7b-philosophes-lora-final.zip -d models/

# Lancer les benchmarks
python scripts/test_model.py --lora ./models/mistral-7b-philosophes-lora-final

# Résultats attendus :
# - 10 questions testées (Spinoza, Bergson, Kant)
# - Accuracy : >80% application correcte schèmes
# - Latence CPU 4-bit : 5-15s par réponse
```

### Comparer avec modèle base (sans LoRA)

```bash
# Tester sans LoRA (baseline)
python scripts/test_model.py --lora None

# Comparer les résultats :
# - LoRA devrait avoir meilleure accuracy sur schèmes
# - Latence similaire (même quantization 4-bit)
```

---

## 📊 Résultats Attendus

### Métriques Training

| Métrique | Valeur Cible |
|----------|--------------|
| **Training loss (final)** | < 0.5 |
| **Eval loss (final)** | < 0.6 |
| **Temps (A100 40GB)** | 30-45 min |
| **Taille LoRA** | 250-350 MB |
| **Paramètres entraînables** | ~1-2% du total |

### Métriques Inférence (CPU 4-bit)

| Métrique | Valeur Cible |
|----------|--------------|
| **Accuracy schèmes** | > 80% |
| **Latence par réponse** | 5-15s (CPU) |
| **Latence par réponse** | 1-3s (GPU) |
| **RAM requise** | ~4 GB (4-bit) |

---

## 🔧 Troubleshooting

### Problème : "CUDA out of memory" sur Colab

**Solution :**
1. Réduire le batch size dans le notebook :
   ```python
   BATCH_SIZE = 2  # Au lieu de 4 ou 8
   GRADIENT_ACCUM = 16  # Compenser
   ```

2. Ou redémarrer le runtime et choisir un GPU plus puissant

### Problème : Training trop lent (>3h sur T4)

**Solution :**
1. Réduire `num_train_epochs` à 2 (au lieu de 3)
2. Ou utiliser seulement `schemes_levelA_base.jsonl` (300 exemples au lieu de 1200)

### Problème : Modèle ne génère que du texte générique

**Solution :**
1. Vérifier que le LoRA a bien été chargé :
   ```python
   model = PeftModel.from_pretrained(model, lora_path)
   ```

2. Vérifier le format du prompt (doit inclure schème + contexte)

3. Vérifier eval_loss : si > 1.0, le training n'a pas convergé (relancer)

---

## 📁 Structure des Fichiers

### Après Training Colab

```
mistral-7b-philosophes-lora-final/
├── adapter_config.json          # Config LoRA
├── adapter_model.safetensors    # Poids LoRA (~250-350 MB)
└── tokenizer*                   # Tokenizer (copie)
```

### Après Download Local

```
/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/
├── models/
│   └── mistral-7b-philosophes-lora-final/  # LoRA téléchargé
├── benchmark_results.json                   # Résultats tests
└── ...
```

---

## 🎯 Prochaines Étapes

### 1. Benchmarks Complets
```bash
# Créer 30 questions test (10 par philosophe)
# Tester accuracy, latence, qualité réponses
python scripts/test_model.py --lora ./models/mistral-7b-philosophes-lora-final
```

### 2. Déploiement HF Space CPU

**Créer un Space HF gratuit :**
1. Créer repo : `FJDaz/mistral-7b-philosophes-demo`
2. Uploader :
   - `app.py` (Gradio interface)
   - `requirements.txt`
   - Charger LoRA depuis HF Hub
3. Hardware : **CPU basic** (gratuit)
4. Tester latence CPU sur Space

### 3. Comparaison avec Qwen 14B SNB

| Critère | Mistral 7B LoRA (CPU) | Qwen 14B LoRA SNB (GPU) |
|---------|----------------------|-------------------------|
| Qualité schèmes | 🟡 Bonne (80-90%) | ✅ Excellente (95%+) |
| Latence | ⚠️ 5-15s (CPU) | ✅ 1-3s (GPU) |
| Coût | ✅ Gratuit (HF CPU) | 💰 Pay-per-use (Modal) |
| Use case | Free tier | Premium tier |

---

## 📞 Support

**Problèmes ou questions ?**
- Consulter `README.md` pour vue d'ensemble
- Consulter `configs/mistral_7b_lora.yaml` pour hyperparams détaillés
- Vérifier `benchmark_results.json` pour métriques attendues

---

**Dernière mise à jour :** 20 novembre 2025
**Auteur :** Claude Code
**Config optimale :** Colab Pro A100 40GB, r=64, batch_size=8, epochs=3
