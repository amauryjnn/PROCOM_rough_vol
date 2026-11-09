"""Fait tourner le CNN sur données factices pour vérifier la boucle."""

from pathlib import Path

import numpy as np
import yaml

from roughvol.donnees_factices import surfaces_factices
from roughvol.entrainement import entrainer
from roughvol.models.cnn import CNN

RACINE = Path(__file__).resolve().parents[1]
N_SURFACES = 2000


def main():
    config = yaml.safe_load((RACINE / "config" / "default.yaml").read_text())
    moneyness = config["grille"]["moneyness"]
    maturites = config["grille"]["maturites"]

    X, y = surfaces_factices(N_SURFACES, moneyness, maturites, config["graine"])

    for padding_mode in ["zeros", "replicate", "reflect"]:
        resultat = entrainer(
            lambda: CNN(len(maturites), len(moneyness), padding_mode=padding_mode),
            X,
            y,
            fractions=config["decoupage"],
            graine=config["graine"],
            **config["entrainement"],
        )
        test = resultat.indices_test.numpy()
        rmse = np.sqrt(((resultat.predire(X[test]).numpy() - y[test]) ** 2).mean(axis=0))
        print(
            f"{padding_mode:10s}"
            f" époques {len(resultat.historique_validation):3d}"
            f" | MSE validation normalisée {resultat.meilleure_validation:.2e}"
            f" | RMSE test par sortie {np.array2string(rmse, precision=3)}"
        )


if __name__ == "__main__":
    main()
