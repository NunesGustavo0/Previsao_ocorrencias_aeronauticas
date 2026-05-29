"""Executa o treinamento mantido em pre_processamento/treino.py."""

from pathlib import Path
import runpy


ROOT_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(ROOT_DIR / "pre_processamento" / "treino.py"), run_name="__main__")
