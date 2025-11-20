# 📋 Synthèse Projet - Fine-Tuning Mistral 7B Philosophes

**Date :** 20 novembre 2025
**Objectif :** Créer une version CPU-compatible de Bergson and Friends pour stratégie freemium

---

## 🎯 Contexte

### Problème Initial
- **HF Spaces GPU coupés** → SNB 14B (Qwen + LoRA Spinoza) impossible sur CPU gratuit
- **Latence CPU prohibitive** → Qwen 14B nécessite GPU (~$18-60/mois ou pay-per-use Modal)

### Solution Proposée
**Fine-tuner Mistral 7B sur schèmes logiques** pour créer un tier gratuit viable :
- Mistral 7B plus efficace sur CPU que Qwen (optimisations internes)
- LoRA spécialisé schèmes logiques → compense taille modèle réduite
- 1200 exemples disponibles (schèmes Niveau A)

---

## 📊 Architecture Freemium

| Critère | Free Tier | Premium Tier |
|---------|-----------|--------------|
| **Modèle** | Mistral 7B + LoRA schèmes | Qwen 14B + LoRA SNB |
| **Infrastructure** | HF Space CPU (gratuit) | Modal GPU A10G (pay-per-use) |
| **Qualité** | 🟡 Bonne (80% accuracy) | ✅ Excellente (95%+ accuracy) |
| **Latence** | ⚠️ 5-15s par réponse | ✅ 1-3s par réponse |
| **Prompts** | Système seul (schèmes dans LoRA) | Système + RAG enrichi |
| **Coût** | ✅ Gratuit | 💰 ~$0.50-1/h utilisation |

**Stratégie :**
- Free tier = **teaser** fonctionnel (attire utilisateurs)
- Premium tier = **qualité + rapidité** (justifie paiement)

---

## 🧬 Dataset Fine-Tuning

### Source
- **Localisation :** `/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/data/FT/processed/`
- **Fichiers :**
  - `schemes_levelA_base.jsonl` (300 exemples)
  - `schemes_levelA_augmented.jsonl` (900 exemples)
- **Total :** 1200 exemples

### Contenu
**Format :** ChatML (system/user/assistant)

**Schèmes logiques couverts :**
- **Modus Ponens** (Si P alors Q, or P, donc Q)
- **Identité** (Dieu = Nature, Liberté = Connaissance nécessité)
- **Causalité** (Tout a une cause, joie → puissance)
- **Opposition** (Durée ≠ temps spatial)
- **Analogie** (Conscience = mélodie)
- **Distinction** (Phénomène ≠ noumène)

**Philosophes :**
- Spinoza : 600 exemples
- Bergson : 300 exemples
- Kant : 300 exemples

**Registre :** Lycéen (Terminale) - vocabulaire accessible, phrases courtes

---

## 🔧 Configuration Training Optimale

### Modèle Base
- **Nom :** `mistralai/Mistral-7B-Instruct-v0.3`
- **Raison :** Optimisé CPU, excellente qualité 7B, support long context

### Méthode Fine-Tuning
- **QLoRA** (4-bit quantization + LoRA)
- **Rang LoRA :** r=64 (haute qualité vs r=8/16 standard)
- **Alpha LoRA :** 128 (= 2 * r)
- **Modules ciblés :** Attention (q/k/v/o) + MLP (gate/up/down)

### Hyperparamètres
- **Batch size effectif :** 32 (8 x 4 grad accum sur A100)
- **Learning rate :** 2e-4 (optimal LoRA)
- **Scheduler :** Cosine avec 3% warmup
- **Epochs :** 3 (évite overfitting sur 1200 exemples)
- **Précision :** bfloat16 (optimal A100)

### Infrastructure
- **GPU optimal :** A100 40GB (Colab Pro)
- **Temps training :** 30-45 minutes
- **Fallback :** V100 16GB (1h-1h30) | T4 15GB (2-3h)

---

## 📦 Livrables Créés

### Documentation
1. **README.md** - Vue d'ensemble projet
2. **QUICKSTART.md** - Guide rapide 3 étapes (5 min setup)
3. **USAGE.md** - Documentation complète (troubleshooting, benchmarks)
4. **PROJECT_SUMMARY.md** - Ce document (synthèse projet)

### Code
1. **notebooks/train_mistral_7b_lora.ipynb** - Notebook Colab clé en main
2. **scripts/test_model.py** - Script benchmarks local (10 questions test)
3. **configs/mistral_7b_lora.yaml** - Configuration hyperparamètres

### Infrastructure
1. **requirements.txt** - Dépendances Python
2. **.gitignore** - Exclusions Git (modèles, cache)
3. **.gitattributes** - Git LFS (si commit checkpoints)

---

## 🎯 Prochaines Étapes

### Immédiat (Vous)
1. ✅ **Upload notebook sur Colab Pro**
2. ✅ **Lancer training** (30-45 min sur A100)
3. ✅ **Download LoRA** (250-350 MB)

### Court Terme (Après Training)
4. **Benchmarks complets** (scripts/test_model.py)
   - 10 questions par philosophe
   - Mesurer accuracy + latence CPU
5. **Comparaison baseline** (Mistral 7B sans LoRA vs avec LoRA)
6. **Déploiement HF Space CPU** (test freemium tier)

### Moyen Terme (Validation Stratégie)
7. **Tests utilisateurs** (échantillon lycéens sur free tier)
8. **Comparaison premium** (Mistral 7B CPU vs Qwen 14B GPU)
9. **Décision finale** freemium strategy

---

## 📊 Résultats Attendus

### Training
- **Training loss :** < 0.5
- **Eval loss :** < 0.6
- **Taille LoRA :** ~250-350 MB
- **Paramètres entraînables :** ~1-2% du total

### Inférence (CPU 4-bit)
- **Accuracy schèmes :** > 80%
- **Latence :** 5-15s par réponse
- **RAM requise :** ~4 GB

### Validation Freemium
- **Free tier viable ?** Oui si latence <15s + accuracy >75%
- **Premium justifié ?** Oui si écart qualité/latence significatif vs free

---

## 🔍 Questions Ouvertes

1. **Mistral 7B LoRA peut-il rivaliser avec prompts système seuls ?**
   → Test ablation : LoRA vs prompts vs LoRA+prompts

2. **Latence CPU acceptable pour lycéens ?**
   → Seuil psychologique : 10-15s (dialogue pédagogique)

3. **Quelle config freemium finale ?**
   - Option A : Free CPU + Premium GPU Modal
   - Option B : Free prompts seuls + Premium LoRA GPU
   - Option C : Pas de free tier, seulement premium abordable

---

## 💡 Insights Stratégiques

### Pourquoi LoRA plutôt que Prompts Seuls ?

**Avantages LoRA :**
- ✅ Application **native** des schèmes (appris dans les poids)
- ✅ Moins de tokens prompt → **latence réduite**
- ✅ Vocabulaire lycéen **internalisé**

**Inconvénient LoRA :**
- ⚠️ Nécessite training (~1h)
- ⚠️ Moins flexible que prompts (modification = re-training)

**Compromis :** LoRA (schèmes) + Prompts système (personnalité philosophe)

### Pourquoi Mistral 7B plutôt que Phi-3 / Qwen 7B ?

| Modèle | Avantages | Inconvénients |
|--------|-----------|---------------|
| **Mistral 7B** | Optimisations CPU, qualité excellente, long context | Pas spécialisé philo |
| Qwen 7B | Bon sur philo, multilingue | Plus lent CPU que Mistral |
| Phi-3.5-mini (3.8B) | Ultra rapide CPU | Qualité moindre (3.8B) |

**Verdict :** Mistral 7B = **meilleur compromis** qualité/performance CPU

---

## 📞 Contact & Support

**Projet :** Bergson and Friends - Freemium Strategy
**Repo :** `/Users/francois-jeandazin/NUX_FT/bergsonAndFriends`
**Auteur :** François-Jean d'Azin
**Assistant :** Claude Code (Anthropic)

**Ressources :**
- Notebook Colab : `notebooks/train_mistral_7b_lora.ipynb`
- Guide rapide : `QUICKSTART.md`
- Documentation : `USAGE.md`
- Config : `configs/mistral_7b_lora.yaml`

---

**Dernière mise à jour :** 20 novembre 2025 - 11:25
**Status :** Setup complet - Prêt pour training Colab Pro
