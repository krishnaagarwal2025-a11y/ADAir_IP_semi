import json
import os
import glob

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "colab": {
                "name": "SEMICON_AdaIR_Training.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in text.split('\n')]
        })

    def add_code(text):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in text.split('\n')]
        })

    # Intro
    add_markdown("# SEMICON AI Hackathon - AdaIR Training\nThis notebook sets up the structure and runs the baseline training.")

    # Setup directories
    add_code(f"!mkdir -p datasets models configs checkpoints results reports adair")

    # Add files
    base_dir = r"C:\Users\varun\.gemini\antigravity\scratch\SEMICON_HACKATHON"
    files = [
        "datasets/semicon_dataset.py",
        "models/adair_semicon.py",
        "configs/train.yaml",
        "train.py",
        "validate.py",
        "inference.py"
    ]

    for rel_path in files:
        full_path = os.path.join(base_dir, rel_path.replace('/', '\\'))
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            cell_content = f"%%writefile {rel_path}\n" + content
            add_code(cell_content)
            print(f"Added {rel_path} to Jupyter Notebook.")

    # Dependencies & Run
    add_markdown("## Setup Dependencies")
    add_code("!pip install pyyaml lpips scikit-image")

    add_markdown("## Run Training")
    add_code("!python train.py --config configs/train.yaml")

    output_path = os.path.join(base_dir, "SEMICON_AdaIR_Colab.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook generated successfully at {output_path}")

if __name__ == "__main__":
    create_notebook()
