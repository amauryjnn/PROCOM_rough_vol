"""Boucle d'entraînement commune aux architectures (FNN / CNN / LSTM).

La boucle ne connaît ni l'origine des données ni le sens du réseau : `X` et `y`
peuvent être (surfaces, paramètres) pour G_w ou (paramètres, surfaces) pour F_w.
"""

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn


def decouper(n, train, validation, test, graine):
    """Répartit les indices 0..n-1 en trois paquets disjoints et reproductibles."""
    if not math.isclose(train + validation + test, 1.0):
        raise ValueError("les fractions du découpage doivent sommer à 1")

    n_train = round(n * train)
    n_validation = round(n * validation)
    if n_train == 0 or n_validation == 0 or n_train + n_validation >= n:
        raise ValueError(f"{n} lignes ne suffisent pas pour ce découpage")

    generateur = torch.Generator().manual_seed(graine)
    indices = torch.randperm(n, generator=generateur)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_validation],
        indices[n_train + n_validation :],
    )


@dataclass
class Normaliseur:
    """Centrage-réduction colonne par colonne."""

    moyenne: torch.Tensor
    ecart_type: torch.Tensor

    @classmethod
    def ajuster(cls, donnees):
        ecart_type = donnees.std(dim=0)
        # Une colonne constante donnerait une division par zéro.
        ecart_type = torch.where(ecart_type > 0, ecart_type, torch.ones_like(ecart_type))
        return cls(donnees.mean(dim=0), ecart_type)

    def appliquer(self, donnees):
        return (donnees - self.moyenne) / self.ecart_type

    def inverser(self, donnees):
        return donnees * self.ecart_type + self.moyenne


@dataclass
class Resultat:
    modele: nn.Module
    norm_x: Normaliseur
    norm_y: Normaliseur
    indices_train: torch.Tensor
    indices_validation: torch.Tensor
    indices_test: torch.Tensor
    graine: int
    historique_train: list
    historique_validation: list

    @property
    def meilleure_validation(self):
        return min(self.historique_validation)

    def predire(self, X):
        """Prédit en unités d'origine."""
        X = torch.as_tensor(X, dtype=torch.float64)
        self.modele.eval()
        with torch.no_grad():
            return self.norm_y.inverser(self.modele(self.norm_x.appliquer(X)))

    def sauvegarder(self, chemin):
        torch.save(
            {
                "poids": self.modele.state_dict(),
                "norm_x": (self.norm_x.moyenne, self.norm_x.ecart_type),
                "norm_y": (self.norm_y.moyenne, self.norm_y.ecart_type),
                "indices_train": self.indices_train,
                "indices_validation": self.indices_validation,
                "indices_test": self.indices_test,
                "graine": self.graine,
                "historique_train": self.historique_train,
                "historique_validation": self.historique_validation,
            },
            chemin,
        )

    @classmethod
    def charger(cls, chemin, fabrique_modele):
        etat = torch.load(chemin, weights_only=True)
        modele = fabrique_modele().double()
        modele.load_state_dict(etat.pop("poids"))
        etat["norm_x"] = Normaliseur(*etat["norm_x"])
        etat["norm_y"] = Normaliseur(*etat["norm_y"])
        return cls(modele=modele, **etat)


def entrainer(
    fabrique_modele,
    X,
    y,
    *,
    fractions,
    graine,
    epoques,
    taille_lot,
    taux_apprentissage,
    patience,
):
    """Entraîne un réseau et renvoie sa meilleure version sur la validation.

    `fabrique_modele` est une fonction qui crée le réseau : il faut le créer
    après avoir fixé la graine pour que l'initialisation soit reproductible.
    `fractions` est le dictionnaire `decoupage` de la configuration.
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)

    train, validation, test = decouper(len(X), **fractions, graine=graine)

    # Statistiques calculées sur le train uniquement : pas de fuite du test.
    norm_x = Normaliseur.ajuster(X[train])
    norm_y = Normaliseur.ajuster(y[train])
    X_norm, y_norm = norm_x.appliquer(X), norm_y.appliquer(y)

    torch.manual_seed(graine)
    generateur_lots = torch.Generator().manual_seed(graine)
    modele = fabrique_modele().double()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=taux_apprentissage)
    critere = nn.MSELoss()

    historique_train, historique_validation = [], []
    meilleurs_poids = copy.deepcopy(modele.state_dict())
    meilleure_perte = math.inf
    epoques_sans_progres = 0

    for _ in range(epoques):
        modele.train()
        ordre = train[torch.randperm(len(train), generator=generateur_lots)]
        perte_cumulee = 0.0
        for lot in ordre.split(taille_lot):
            optimiseur.zero_grad()
            perte = critere(modele(X_norm[lot]), y_norm[lot])
            perte.backward()
            optimiseur.step()
            perte_cumulee += perte.item() * len(lot)
        historique_train.append(perte_cumulee / len(train))

        modele.eval()
        with torch.no_grad():
            perte_validation = critere(modele(X_norm[validation]), y_norm[validation]).item()
        historique_validation.append(perte_validation)

        if perte_validation < meilleure_perte:
            meilleure_perte = perte_validation
            meilleurs_poids = copy.deepcopy(modele.state_dict())
            epoques_sans_progres = 0
        else:
            epoques_sans_progres += 1
            if epoques_sans_progres >= patience:
                break

    modele.load_state_dict(meilleurs_poids)
    return Resultat(
        modele=modele,
        norm_x=norm_x,
        norm_y=norm_y,
        indices_train=train,
        indices_validation=validation,
        indices_test=test,
        graine=graine,
        historique_train=historique_train,
        historique_validation=historique_validation,
    )
