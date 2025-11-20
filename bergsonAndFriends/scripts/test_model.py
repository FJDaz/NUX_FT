"""
Script de test du modèle Mistral 7B LoRA fine-tuné
Teste l'application des schèmes logiques sur 10 questions par philosophe
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import time
from typing import List, Dict

# Configuration
MODEL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_PATH = "./models/mistral-7b-philosophes-lora-final"  # Chemin vers LoRA

# Questions de test par philosophe
TEST_QUESTIONS = {
    "spinoza": [
        {
            "schema": "Modus Ponens",
            "context": "Si l'homme ignore les causes de ses passions, il est en servitude. Or l'élève ignore les causes de ses passions.",
            "expected": "Donc l'élève est en servitude."
        },
        {
            "schema": "Identité",
            "context": "Dieu = Nature. Or la Nature est nécessaire.",
            "expected": "Donc Dieu est nécessaire."
        },
        {
            "schema": "Causalité",
            "context": "La joie augmente la puissance d'agir. Or comprendre ses affects produit de la joie.",
            "expected": "Donc comprendre ses affects augmente la puissance d'agir."
        },
    ],
    "bergson": [
        {
            "schema": "Opposition",
            "context": "La durée est vécue de l'intérieur. Le temps spatial est mesuré de l'extérieur.",
            "expected": "Donc la durée s'oppose au temps spatial."
        },
        {
            "schema": "Analogie",
            "context": "La conscience est comme une mélodie : chaque note s'interpénètre avec les autres.",
            "expected": "Donc la conscience est une continuité fluide (comme une mélodie)."
        },
    ],
    "kant": [
        {
            "schema": "Distinction",
            "context": "Le phénomène est ce qui apparaît. Le noumène est la chose en soi.",
            "expected": "Donc phénomène ≠ noumène."
        },
        {
            "schema": "Condition",
            "context": "Pour être autonome, il faut suivre une loi qu'on se donne à soi-même.",
            "expected": "Donc l'autonomie implique l'auto-législation."
        },
    ],
}


def load_model(lora_path: str = None, device: str = "auto"):
    """
    Charge le modèle Mistral 7B avec ou sans LoRA

    Args:
        lora_path: Chemin vers LoRA (None = base model)
        device: "auto", "cpu", "cuda"
    """
    print(f"📥 Chargement du modèle base: {MODEL_BASE}")

    # Configuration quantization 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    # Charger le modèle base
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Charger le LoRA si fourni
    if lora_path:
        print(f"📥 Chargement du LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge pour inférence
        print("✅ LoRA chargé et mergé")
    else:
        print("⚠️ Mode BASE (sans LoRA)")

    return model, tokenizer


def test_schema(model, tokenizer, philosopher: str, question: Dict, verbose: bool = True):
    """
    Teste l'application d'un schème logique

    Returns:
        dict avec résultats (response, latency, correct)
    """
    # Construire le prompt
    prompt = f"Schème : {question['schema']}\nContexte : {question['context']}\nApplique le schème :"

    messages = [
        {"role": "system", "content": "Tu es un tuteur philosophique maîtrisant les schèmes logiques. Tu appliques le schème demandé au contexte fourni."},
        {"role": "user", "content": prompt}
    ]

    # Tokeniser
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Générer
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.time() - start_time

    # Décoder
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    # Vérifier si correct (simple check si conclusion présente)
    correct = question['expected'].lower() in response.lower()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Philosophe: {philosopher.upper()}")
        print(f"Schème: {question['schema']}")
        print(f"Contexte: {question['context']}")
        print(f"\nAttendu: {question['expected']}")
        print(f"Réponse: {response}")
        print(f"Latence: {latency:.2f}s")
        print(f"Correct: {'✅' if correct else '❌'}")

    return {
        "philosopher": philosopher,
        "schema": question['schema'],
        "response": response,
        "latency": latency,
        "correct": correct,
    }


def run_benchmarks(model, tokenizer, save_results: bool = True):
    """
    Lance tous les benchmarks et affiche les résultats
    """
    print("\n" + "="*60)
    print("🎯 BENCHMARKS - Application Schèmes Logiques")
    print("="*60)

    all_results = []

    # Tester chaque philosophe
    for philosopher, questions in TEST_QUESTIONS.items():
        print(f"\n📚 Test {philosopher.upper()} ({len(questions)} questions)")

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]", end=" ")
            result = test_schema(model, tokenizer, philosopher, question)
            all_results.append(result)

    # Statistiques globales
    print("\n" + "="*60)
    print("📊 STATISTIQUES GLOBALES")
    print("="*60)

    total = len(all_results)
    correct = sum(1 for r in all_results if r['correct'])
    avg_latency = sum(r['latency'] for r in all_results) / total

    print(f"Total questions: {total}")
    print(f"Réponses correctes: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Latence moyenne: {avg_latency:.2f}s")

    # Par philosophe
    print("\nPar philosophe:")
    for philosopher in TEST_QUESTIONS.keys():
        phil_results = [r for r in all_results if r['philosopher'] == philosopher]
        phil_correct = sum(1 for r in phil_results if r['correct'])
        phil_latency = sum(r['latency'] for r in phil_results) / len(phil_results)
        print(f"  {philosopher.upper()}: {phil_correct}/{len(phil_results)} correct, {phil_latency:.2f}s avg")

    # Sauvegarder les résultats
    if save_results:
        output_file = "./benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total,
                    "avg_latency": avg_latency,
                },
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Résultats sauvegardés: {output_file}")

    return all_results


def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Mistral 7B LoRA")
    parser.add_argument("--lora", type=str, default=LORA_PATH, help="Chemin vers LoRA (None = base model)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder les résultats")

    args = parser.parse_args()

    # Vérifier GPU
    if torch.cuda.is_available():
        print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ Pas de GPU, utilisation CPU (latence élevée attendue)")

    # Charger le modèle
    model, tokenizer = load_model(args.lora, args.device)

    # Lancer les benchmarks
    results = run_benchmarks(model, tokenizer, save_results=not args.no_save)

    print("\n✅ Tests terminés !")


if __name__ == "__main__":
    main()
