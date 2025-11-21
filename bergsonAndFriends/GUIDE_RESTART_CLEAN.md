# 🚀 Guide Restart Propre - Mistral 7B Fine-Tuning

**Date :** 20 novembre 2025
**Objectif :** Relancer le fine-tuning proprement avec dataset combiné (80% schèmes + 20% incarnation)

---

## 📦 Fichiers à Préparer

### 1. Notebook propre
✅ **Créé** : `/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/notebooks/train_mistral_7b_lora_CLEAN.ipynb`

### 2. Datasets à uploader
Localisation : `/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/data/FT/processed/`

```bash
# Vérifier que les fichiers existent
ls -lh /Users/francois-jeandazin/NUX_FT/bergsonAndFriends/data/FT/processed/*.jsonl
```

**Fichiers requis :**
- ✅ `schemes_levelA_base.jsonl` (~545 KB, 300 exemples)
- ✅ `schemes_levelA_augmented.jsonl` (~1.6 MB, 900 exemples)
- ✅ `enriched_correction_dataset.jsonl` (~100 KB, 213 exemples)

---

## 🎯 Plan d'Action (5 étapes)

### ÉTAPE 1 : Nettoyer Colab (2 min)

1. Si Colab est ouvert : **Runtime > Disconnect and delete runtime**
2. Fermer l'onglet Colab
3. Attendre 30 secondes

### ÉTAPE 2 : Upload notebook (1 min)

1. Aller sur [Google Colab](https://colab.research.google.com/)
2. **File > Upload notebook**
3. Sélectionner : `/Users/francois-jeandazin/NUX_FT/bergsonAndFriends/notebooks/train_mistral_7b_lora_CLEAN.ipynb`
4. **Runtime > Change runtime type > GPU = L4** (ou A100 si dispo)

### ÉTAPE 3 : Upload datasets (2 min)

Dans Colab, panneau **Files** (icône 📁 à gauche) :

1. Cliquer sur **Upload to session storage** (icône upload)
2. Sélectionner les 3 fichiers JSONL :
   - `schemes_levelA_base.jsonl`
   - `schemes_levelA_augmented.jsonl`
   - `enriched_correction_dataset.jsonl`
3. Attendre que les 3 fichiers apparaissent dans `/content/`

**Vérification :**
```python
!ls -lh /content/*.jsonl
```

Doit afficher :
```
-rw-r--r-- 1 root root 545K enriched_correction_dataset.jsonl
-rw-r--r-- 1 root root 100K schemes_levelA_augmented.jsonl
-rw-r--r-- 1 root root 1.6M schemes_levelA_base.jsonl
```

### ÉTAPE 4 : Configurer token HF (1 min)

**Option A : Colab Secrets (RECOMMANDÉ)**

1. Cliquer sur l'icône **🔑** (Secrets) dans la barre gauche
2. Cliquer **+ Add new secret**
3. Name: `HF_TOKEN`
4. Value: `hf_...` (votre token HF avec write access)
5. Toggle : **Activer l'accès** pour ce notebook

**Option B : Saisie manuelle**

Le notebook vous demandera le token si Secrets non configuré.

### ÉTAPE 5 : Lancer le training (3-4h total)

**Exécution séquentielle :**

```
Section 1  : Installation packages        [3 min]
Section 2  : Authentification HF          [10 sec]
Section 3  : Chargement datasets          [30 sec]
Section 4  : Chargement modèle + LoRA     [10 min]
Section 5  : Training INITIAL             [2-3h sur L4] ⚠️ LONG
Section 6  : Sauvegarde checkpoint        [1 min]
Section 7  : Test schèmes                 [30 sec]
Section 8  : Re-fine-tuning COMBINÉ       [1-1.5h sur L4]
Section 9  : Test dialogue interactif     [illimité]
```

**Comment lancer :**

- **Option rapide** : `Runtime > Run all` (lance tout d'un coup)
- **Option prudente** : Exécuter cellule par cellule (Shift+Enter)

---

## ⏱️ Temps Estimés par GPU

| GPU | Section 5 (initial) | Section 8 (combiné) | **TOTAL** |
|-----|---------------------|---------------------|-----------|
| **A100 40GB** | 30-45 min | 30 min | **~1h** |
| **V100 16GB** | 1h-1h30 | 45 min | **~2h15** |
| **L4 15GB** | 2-3h | 1-1.5h | **~3-4h** |
| **T4 15GB** | 2-3h | 1-1.5h | **~3-4h** |

---

## 📊 Monitoring du Training

### Métriques à surveiller

**Section 5 (initial) :**
```
Step 50  : loss=1.234, eval_loss=1.567
Step 100 : loss=0.890, eval_loss=1.123
...
Step 300 : loss=0.456, eval_loss=0.678  ← Objectif final
```

**Objectifs :**
- ✅ `eval_loss` < 0.6 (validation OK)
- ✅ `eval_loss` diminue progressivement
- ⚠️ Si `eval_loss` remonte → overfitting (mais load_best_model_at_end gère)

**Section 8 (combiné) :**
- Mêmes objectifs
- Surveillance toutes les 20 steps (plus fréquent)

---

## ✅ Vérifications Post-Training

### Après Section 7 (test schèmes)

**Attendu :**
```
Réponse: Donc l'élève est en servitude.
```

**Problème si :**
- Répond hors sujet
- N'applique pas le schème
- Génère du charabia

### Après Section 9 (dialogue interactif)

**Tests à faire :**

1. **Test 1ère personne :**
   ```
   VOUS : Ben, c'est pas toi Spinoza ?
   SPINOZA : Oui, je suis Spinoza. Je te parle en première personne. [OK]
   SPINOZA : Oui, c'est Spinoza. Pour Spinoza, ... [❌ 3ème personne]
   ```

2. **Test schèmes :**
   ```
   VOUS : La liberté, c'est faire ce qu'on veut ?
   SPINOZA : Pas vraiment. Mais alors, si tu fais ce que tu veux sans comprendre pourquoi... [OK]
   ```

3. **Test répétition :**
   - Poser 3 questions différentes
   - Vérifier que les réponses varient

---

## 🎯 Résultats Attendus

### Checkpoint final

**Localisation Colab :**
- `/content/mistral-combined-final/` (250-350 MB)

**Fichiers :**
```
mistral-combined-final/
├── adapter_config.json
├── adapter_model.safetensors  (~300 MB)
└── tokenizer files
```

### Push Hugging Face

**URL :** `https://huggingface.co/spaces/FJDaz/3_PHI/tree/main/Spinoza_Secours`

**Structure finale HF Space :**
```
FJDaz/3_PHI/
├── qwen-spinoza-niveau-b/   ← LoRA SNB original (INCHANGÉ)
└── Spinoza_Secours/         ← LoRA Mistral 7B nouveau
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## ⚠️ Troubleshooting

### Erreur : "FileNotFoundError: schemes_levelA_base.jsonl"

**Cause :** Fichiers JSONL non uploadés dans Colab

**Solution :**
1. Vérifier panneau Files (gauche)
2. Re-uploader les 3 fichiers JSONL
3. Relancer la cellule

### Erreur : "RuntimeError: CUDA out of memory"

**Cause :** GPU trop petit ou batch size trop grand

**Solution :**
```python
# Dans une nouvelle cellule, avant Section 5
BATCH_SIZE = 1
GRADIENT_ACCUM = 32
```

Puis relancer les sections 5 et 8.

### Training bloqué (pas de progression)

**Symptômes :**
- Cellule tourne depuis >20 min
- Pas de barres de progression
- Pas de logs

**Solution :**
1. **Stop** (carré à gauche de la cellule)
2. Si bloqué : `Runtime > Interrupt execution`
3. Si encore bloqué : `Runtime > Disconnect and delete runtime`
4. Recommencer depuis ÉTAPE 1

### eval_loss > 1.0 (ne converge pas)

**Cause :** Learning rate trop élevé ou dataset corrompu

**Solution :**
```python
# Modifier dans Section 5 et 8
learning_rate=1e-4,  # Au lieu de 2e-4
```

---

## 📞 Support

**Documentation complète :**
- [README.md](README.md) - Vue d'ensemble
- [QUICKSTART.md](QUICKSTART.md) - Guide rapide
- [USAGE.md](USAGE.md) - Guide détaillé
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problèmes courants
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Synthèse projet

**En cas de blocage :**
1. Vérifier [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Copier l'erreur complète
3. Demander à Claude Code

---

## 🎓 Après le Training

### Télécharger le modèle localement

```python
# Dernière cellule du notebook
!zip -r mistral-combined-final.zip /content/mistral-combined-final/
```

Puis : **Files > Download** `mistral-combined-final.zip`

### Tester localement

```bash
cd /Users/francois-jeandazin/NUX_FT/bergsonAndFriends
unzip mistral-combined-final.zip

python scripts/test_model.py --lora ./mistral-combined-final
```

---

**Dernière mise à jour :** 20 novembre 2025 - 14:30
**Version notebook :** CLEAN v2
**Status :** ✅ Prêt pour training
