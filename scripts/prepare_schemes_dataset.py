#!/usr/bin/env python3
"""
Prépare le dataset "Dataset Niveau A Schemes.txt" pour un fine-tuning LoRA.

Fonctionnalités :
1. Nettoyage d'encodage (corrige les "Ã©" etc.)
2. Extraction des métadonnées (schème, contexte, difficulté heuristique)
3. Génération d'un JSONL "base" + d'un JSONL "augmenté" avec variations de registre
4. Résumé des statistiques par schème

Usage :
    python scripts/prepare_schemes_dataset.py \
        --input data/FT/Dataset\ Niveau\ A\ Schemes.txt \
        --output-dir data/FT/processed
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_INPUT = Path("data/FT/Dataset Niveau A Schemes.txt")
DEFAULT_OUTPUT_DIR = Path("data/FT/processed")

BASE_SYSTEM_PROMPT = (
    "Tu es un tuteur philosophique maîtrisant les schèmes logiques. "
    "Tu appliques le schème demandé au contexte fourni. "
    "Explique toujours en langage clair et rigoureux."
)

REGISTER_INSTRUCTIONS = {
    "lyceen": (
        "Adapte-toi à un élève de Terminale : phrases courtes, vocabulaire accessible, "
        "exemples concrets. Explique les termes techniques."
    ),
    "intermediaire": (
        "Langage clair mais structuré. Tu peux utiliser quelques termes techniques "
        "à condition de les relier à des exemples."
    ),
    "avance": (
        "Autorise un registre plus soutenu et des distinctions fines. "
        "Tu peux mobiliser des concepts abstraits sans tout détailler."
    ),
}

SYSTEM_SUFFIX_BY_REGISTER = {
    reg: f"{BASE_SYSTEM_PROMPT}\n\nADAPTATION REGISTRE ({reg.upper()}) : {instruction}"
    for reg, instruction in REGISTER_INSTRUCTIONS.items()
}


@dataclass
class Example:
    schema: str
    context: str
    user_prompt: str
    assistant: str
    level: str = "A"

    def to_record(self, register: str = "lyceen") -> Dict:
        system_prompt = SYSTEM_SUFFIX_BY_REGISTER.get(register, BASE_SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.user_prompt},
            {"role": "assistant", "content": self.assistant},
        ]
        difficulty = classify_difficulty(self.context)
        return {
            "schema": self.schema,
            "context": self.context,
            "level": self.level,
            "register": register,
            "difficulty": difficulty,
            "system": system_prompt,
            "user": self.user_prompt,
            "assistant": self.assistant,
            "messages": messages,
            "metadata": {
                "schema": self.schema,
                "register": register,
                "difficulty": difficulty,
            },
        }


def main() -> None:
    args = parse_args()
    raw_examples = load_raw_examples(args.input)
    processed = [transform_example(item) for item in raw_examples]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = output_dir / "schemes_levelA_base.jsonl"
    augmented_path = output_dir / "schemes_levelA_augmented.jsonl"

    write_jsonl(base_path, (ex.to_record("lyceen") for ex in processed))
    write_jsonl(
        augmented_path,
        (
            ex.to_record(register)
            for ex in processed
            for register in REGISTER_INSTRUCTIONS.keys()
        ),
    )

    summarize(processed, base_path, augmented_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoie et prépare le dataset Schemes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Chemin du dataset brut.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie (créé si besoin).",
    )
    return parser.parse_args()


def load_raw_examples(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Impossible de parser {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Le dataset doit être une liste JSON.")
    return data


def transform_example(raw: dict) -> Example:
    messages = raw.get("messages") or []
    if len(messages) < 3:
        raise ValueError("Chaque exemple doit contenir system/user/assistant.")

    user = normalize_text(messages[1]["content"])
    assistant = normalize_text(messages[2]["content"])

    schema = extract_schema(user)
    context = extract_context(user)

    return Example(
        schema=schema,
        context=context,
        user_prompt=user,
        assistant=assistant,
    )


def normalize_text(text: str) -> str:
    """
    Corrige les artefacts d'encodage type 'SchÃ¨me' -> 'Schème' et uniformise en NFC.
    """
    if not text:
        return ""
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n")
    return text.strip()


def extract_schema(user_prompt: str) -> str:
    match = re.search(r"Sch[èe]me\s*:\s*(.+)", user_prompt, flags=re.IGNORECASE)
    raw_schema = match.group(1).strip() if match else "inconnu"
    schema_slug = re.sub(r"[^a-z0-9]+", "_", raw_schema.lower()).strip("_")
    return schema_slug or "inconnu"


def extract_context(user_prompt: str) -> str:
    match = re.search(r"Contexte\s*:\s*(.+)", user_prompt, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def classify_difficulty(context: str) -> str:
    length = len(context.split())
    if length < 15:
        return "facile"
    if length < 35:
        return "intermediaire"
    return "avance"


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(examples: List[Example], base_path: Path, augmented_path: Path) -> None:
    counter = Counter(ex.schema for ex in examples)
    total = len(examples)
    print(f"✅ {total} exemples transformés.")
    for schema, count in counter.most_common():
        pct = count / total * 100 if total else 0
        print(f"   - {schema:<20} : {count:>4} ({pct:4.1f}%)")
    print(f"\n📄 Sorties :\n   • {base_path}\n   • {augmented_path}")


if __name__ == "__main__":
    main()

