# 🎓 Fine-Tuning Mistral 7B - Philosophes (Schèmes Logiques)

**Objectif :** Fine-tuner Mistral 7B sur 1200 exemples de schèmes logiques philosophiques pour créer une version CPU-compatible du système Bergson and Friends.

## 📊 Datasets

- **Base :** 300 exemples (`data/FT/processed/schemes_levelA_base.jsonl`)
- **Augmentés :** 900 exemples (`data/FT/processed/schemes_levelA_augmented.jsonl`)
- **Total :** 1200 exemples de schèmes logiques (Modus Ponens, identités spinozistes, causalité)
- **Format :** ChatML (system/user/assistant)
- **Registre :** Lycéen (Terminale)

## 🎯 Stratégie

### Modèle Base
- **Mistral 7B Instruct v0.3** (`mistralai/Mistral-7B-Instruct-v0.3`)
- Fine-tuning : **QLoRA** (4-bit quantization)
- Méthode : **PEFT/LoRA** (rang 64, alpha 128)

### Use Case
- **Free Tier :** Mistral 7B LoRA sur CPU HF Space (latence 5-15s)
- **Premium Tier :** Qwen 14B LoRA SNB sur GPU Modal (latence <2s)

### Qualité Attendue
- Application rigoureuse des schèmes logiques
- Réponses structurées (prémisses → conclusion)
- Vocabulaire lycéen accessible

## 🚀 Training

### Colab Pro (Recommandé)
```bash
# Ouvrir le notebook Colab
notebooks/train_mistral_7b_lora.ipynb

# GPU optimal : A100 40GB
# Temps : 30-45 minutes
# Config : r=64, batch_size=8, epochs=3
```

### Local (Backup)
```bash
# Avec MPS (Apple Silicon) ou CPU
python scripts/train_local.py --config configs/mistral_7b_lora.yaml
```

## 📁 Structure

```
/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/
├── data/
│   └── FT/
│       ├── Dataset Niveau A Schemes.txt
│       └── processed/
│           ├── schemes_levelA_base.jsonl (300 ex)
│           └── schemes_levelA_augmented.jsonl (900 ex)
├── notebooks/
│   └── train_mistral_7b_lora.ipynb (Colab Pro)
├── scripts/
│   ├── train_local.py (training local)
│   └── test_model.py (benchmarks)
├── configs/
│   └── mistral_7b_lora.yaml (hyperparams)
└── models/
    └── (LoRA checkpoints après training)
```

## 🎓 Philosophes Cibles

- **Spinoza :** Schèmes d'identité (Dieu=Nature), causalité nécessaire, affects
- **Bergson :** Opposition durée/temps spatial, métaphores temporelles
- **Kant :** Distinctions (phénomène/noumène), conditions transcendantales

## 📈 Benchmarks

Après training, tester sur 10 questions par philosophe :
- Application correcte des schèmes logiques
- Cohérence style conversationnel lycéen
- Latence CPU (4-bit quantization)

---

**Dernière mise à jour :** 20 novembre 2025
