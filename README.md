# NUX_FT - Fine-Tuning pour Bergson and Friends

Repo dédié au fine-tuning de modèles (Mistral 7B + Schemes) pour le projet Bergson and Friends.

## Structure

```
NUX_FT/
├── bergsonAndFriends/     # Scripts et datasets depuis le repo BAF principal
├── notebooks/             # Notebooks Colab pour fine-tuning
├── checkpoints/           # LoRA adapters (gitignored)
└── scripts/               # Scripts utilitaires
```

## Usage

1. Cloner ce repo dans Colab
2. Charger le dataset depuis `bergsonAndFriends/data/FT/processed/`
3. Lancer le fine-tuning LoRA

