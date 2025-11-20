# ⚡ Quick Start - Fine-Tuning Mistral 7B

## 🎯 Objectif
Fine-tuner Mistral 7B sur 1200 exemples de schèmes logiques philosophiques pour créer une version CPU-compatible (freemium tier) de Bergson and Friends.

---

## 🚀 3 Étapes pour Lancer le Training

### 1️⃣ Préparer Colab Pro (5 min)

1. Ouvrir [Google Colab](https://colab.research.google.com/)
2. **Upload :** `notebooks/train_mistral_7b_lora.ipynb`
3. **Sélectionner GPU :** Runtime > Change runtime type > **A100 GPU** ⚡
4. **Upload datasets :**
   - Depuis Colab : Files (icône gauche) > Upload
   - Uploader les 2 fichiers :
     - `data/FT/processed/schemes_levelA_base.jsonl` (545 KB)
     - `data/FT/processed/schemes_levelA_augmented.jsonl` (1.6 MB)

### 2️⃣ Configurer le Notebook (2 min)

Dans la cellule **"2️⃣ Configuration"**, remplacer :

```python
# Votre token Hugging Face (avec write access)
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

# Votre username Hugging Face
HF_USERNAME = "FJDaz"  # Remplacer par votre username
```

**Obtenir un token HF :**
1. Aller sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Créer un token avec **write access**
3. Copier le token dans le notebook

### 3️⃣ Lancer le Training (30-45 min sur A100)

1. **Run All :** Runtime > Run all (ou Ctrl+F9)
2. ☕ **Pause café** : 30-45 min sur A100 40GB
3. ✅ **Récupérer le modèle :**
   - **Option A :** Download `.zip` depuis Colab
   - **Option B :** Auto-push sur `huggingface.co/FJDaz/mistral-7b-philosophes-lora`

---

## 📊 Résultats Attendus

### Métriques Training
- **Training loss (final) :** < 0.5
- **Eval loss (final) :** < 0.6
- **Temps A100 40GB :** 30-45 minutes ⚡
- **Taille LoRA :** ~250-350 MB

### Métriques Inférence (après tests)
- **Accuracy schèmes :** > 80%
- **Latence CPU 4-bit :** 5-15s par réponse
- **RAM requise :** ~4 GB

---

## 🧪 Test Rapide du Modèle (Local)

### Installer les dépendances
```bash
cd /Users/francois-jeandazin/NUX_FT/bergsonAndFriends
pip install -r requirements.txt
```

### Dézipper et tester le LoRA
```bash
# Dézipper le modèle téléchargé depuis Colab
unzip mistral-7b-philosophes-lora-final.zip -d models/

# Lancer les benchmarks
python scripts/test_model.py --lora ./models/mistral-7b-philosophes-lora-final
```

**Output attendu :**
```
🎯 BENCHMARKS - Application Schèmes Logiques
===========================================================
📚 Test SPINOZA (3 questions)
[1/3]
Philosophe: SPINOZA
Schème: Modus Ponens
Réponse: Donc l'élève est en servitude.
Latence: 8.2s
Correct: ✅
...

📊 STATISTIQUES GLOBALES
===========================================================
Total questions: 10
Réponses correctes: 8/10 (80.0%)
Latence moyenne: 7.5s
```

---

## 🎓 Philosophes Couverts

**Dataset :** 1200 exemples de schèmes logiques

| Philosophe | Schèmes Principaux | Exemples |
|------------|-------------------|----------|
| **Spinoza** | Modus Ponens, Identité (Dieu=Nature), Causalité | 600 ex |
| **Bergson** | Opposition (durée/temps), Analogie (mélodie) | 300 ex |
| **Kant** | Distinction (phénomène/noumène), Condition | 300 ex |

---

## 📈 Comparaison Freemium Strategy

| Tier | Modèle | Infra | Qualité | Latence | Coût |
|------|--------|-------|---------|---------|------|
| **Free** | Mistral 7B LoRA (CPU) | HF Space gratuit | 🟡 Bonne (80%) | ⚠️ 5-15s | ✅ Gratuit |
| **Premium** | Qwen 14B LoRA SNB (GPU) | Modal/HF GPU | ✅ Excellente (95%+) | ✅ 1-3s | 💰 Pay-per-use |

**Objectif :** Valider que Mistral 7B LoRA (free tier) est **suffisamment bon** pour attirer des utilisateurs, tout en justifiant le premium (latence + qualité).

---

## 🔗 Ressources

- **Documentation complète :** [USAGE.md](USAGE.md)
- **Architecture :** [README.md](README.md)
- **Config détaillée :** [configs/mistral_7b_lora.yaml](configs/mistral_7b_lora.yaml)
- **Notebook Colab :** [notebooks/train_mistral_7b_lora.ipynb](notebooks/train_mistral_7b_lora.ipynb)

---

## 💡 Prochaines Étapes

1. ✅ **Training terminé** → Passer aux benchmarks complets
2. 📊 **Benchmarks OK** → Déployer sur HF Space CPU gratuit
3. 🎯 **Space deployed** → Comparer avec Qwen 14B SNB (premium)
4. 🚀 **Comparaison faite** → Décider stratégie freemium finale

---

**Créé le :** 20 novembre 2025
**Config optimale :** Colab Pro A100 40GB, LoRA r=64, batch_size=8, epochs=3
**Temps total :** ~1h (setup 5min + training 45min + tests 10min)
