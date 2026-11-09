"""Surfaces factices pour vérifier la boucle d'entraînement.

Ce ne sont pas des surfaces rough Heston et les 6 paramètres n'ont pas le sens
de (H, V0, theta, kappa, nu, rho). Aucun résultat obtenu ici ne dit quoi que ce
soit sur la calibration.
"""

import numpy as np


def surfaces_factices(n, moneyness, maturites, graine):
    """Renvoie X (n x 54, maturité externe / moneyness interne) et y (n x 6)."""
    rng = np.random.default_rng(graine)
    y = rng.uniform(-1.0, 1.0, size=(n, 6))

    t, k = np.meshgrid(
        np.asarray(maturites, dtype=np.float64),
        np.asarray(moneyness, dtype=np.float64) - 1.0,
        indexing="ij",
    )
    # Six fonctions indépendantes : les paramètres sont identifiables.
    base = np.stack([np.ones_like(k), k, k**2, t, t * k, t * k**2], axis=-1)
    X = y @ base.reshape(-1, 6).T
    return X, y
