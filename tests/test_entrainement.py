import numpy as np
import pytest
import torch
from torch import nn

from roughvol.donnees_factices import surfaces_factices
from roughvol.entrainement import Normaliseur, Resultat, decouper, entrainer
from roughvol.models.cnn import CNN

MONEYNESS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
MATURITES = [0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REGLAGES = dict(fractions=FRACTIONS, graine=0, taille_lot=16, taux_apprentissage=1e-3)


def lineaire():
    return nn.Linear(54, 6)


@pytest.fixture
def donnees():
    return surfaces_factices(100, MONEYNESS, MATURITES, graine=0)


def test_decoupage_disjoint_complet_et_aux_bonnes_proportions():
    train, validation, test = decouper(1000, **FRACTIONS, graine=0)
    assert (len(train), len(validation), len(test)) == (700, 150, 150)
    assert sorted(torch.cat([train, validation, test]).tolist()) == list(range(1000))


def test_decoupage_reproductible_et_dependant_de_la_graine():
    premier = decouper(100, **FRACTIONS, graine=1)[0]
    assert torch.equal(premier, decouper(100, **FRACTIONS, graine=1)[0])
    assert not torch.equal(premier, decouper(100, **FRACTIONS, graine=2)[0])


def test_decoupage_refuse_des_fractions_incoherentes():
    with pytest.raises(ValueError):
        decouper(100, train=0.85, validation=0.15, test=0.15, graine=0)


def test_normaliseur_aller_retour_et_colonne_constante():
    donnees_brutes = torch.tensor([[1.0, 5.0], [3.0, 5.0], [8.0, 5.0]], dtype=torch.float64)
    normaliseur = Normaliseur.ajuster(donnees_brutes)
    assert normaliseur.ecart_type[1] == 1.0
    assert torch.allclose(normaliseur.inverser(normaliseur.appliquer(donnees_brutes)), donnees_brutes)


def test_normalisation_ne_voit_que_le_train(donnees):
    X, y = donnees
    reference = entrainer(lineaire, X, y, epoques=1, patience=1, **REGLAGES)

    X_modifie, y_modifie = X.copy(), y.copy()
    hors_train = torch.cat([reference.indices_validation, reference.indices_test]).numpy()
    X_modifie[hors_train] += 1000.0
    y_modifie[hors_train] += 1000.0
    modifie = entrainer(lineaire, X_modifie, y_modifie, epoques=1, patience=1, **REGLAGES)

    assert torch.equal(reference.norm_x.moyenne, modifie.norm_x.moyenne)
    assert torch.equal(reference.norm_x.ecart_type, modifie.norm_x.ecart_type)
    assert torch.equal(reference.norm_y.moyenne, modifie.norm_y.moyenne)
    assert torch.equal(reference.norm_y.ecart_type, modifie.norm_y.ecart_type)


def test_entrainement_reproductible(donnees):
    X, y = donnees
    premier = entrainer(lineaire, X, y, epoques=5, patience=5, **REGLAGES)
    second = entrainer(lineaire, X, y, epoques=5, patience=5, **REGLAGES)
    assert premier.historique_validation == second.historique_validation


def test_tout_est_en_float64(donnees):
    X, y = donnees
    resultat = entrainer(lineaire, X, y, epoques=1, patience=1, **REGLAGES)
    assert all(poids.dtype == torch.float64 for poids in resultat.modele.parameters())
    assert resultat.predire(X).dtype == torch.float64


def test_garde_le_meilleur_modele_sur_la_validation(donnees):
    X, y = donnees
    # Un taux énorme fait diverger : le dernier état n'est pas le meilleur.
    reglages = {**REGLAGES, "taux_apprentissage": 10.0}
    resultat = entrainer(lineaire, X, y, epoques=20, patience=20, **reglages)
    assert resultat.historique_validation[-1] > resultat.meilleure_validation

    validation = resultat.indices_validation
    predictions = resultat.norm_y.appliquer(resultat.predire(X[validation]))
    cibles = resultat.norm_y.appliquer(torch.as_tensor(y[validation]))
    assert ((predictions - cibles) ** 2).mean().item() == pytest.approx(resultat.meilleure_validation)


def test_arret_quand_la_validation_ne_progresse_plus(donnees):
    X, y = donnees
    reglages = {**REGLAGES, "taux_apprentissage": 0.0}
    resultat = entrainer(lineaire, X, y, epoques=50, patience=3, **reglages)
    assert len(resultat.historique_validation) == 4


def test_sauvegarde_puis_rechargement(donnees, tmp_path):
    X, y = donnees
    resultat = entrainer(lineaire, X, y, epoques=2, patience=2, **REGLAGES)
    chemin = tmp_path / "modele.pt"
    resultat.sauvegarder(chemin)
    recharge = Resultat.charger(chemin, lineaire)
    assert torch.equal(resultat.predire(X), recharge.predire(X))
    assert torch.equal(resultat.indices_test, recharge.indices_test)
    assert recharge.graine == resultat.graine


def test_ordre_des_surfaces_factices():
    X, y = surfaces_factices(3, MONEYNESS, MATURITES, graine=0)
    assert X.shape == (3, 54) and y.shape == (3, 6)
    assert X.dtype == np.float64 and y.dtype == np.float64

    # Maturité externe, moneyness interne : ligne 1, colonne 2 de la grille.
    indice = 1 * len(MONEYNESS) + 2
    t, k = MATURITES[1], MONEYNESS[2] - 1.0
    p = y[0]
    attendu = p[0] + p[1] * k + p[2] * k**2 + t * (p[3] + p[4] * k + p[5] * k**2)
    assert X[0, indice] == pytest.approx(attendu)


def test_le_cnn_fait_mieux_que_predire_la_moyenne():
    X, y = surfaces_factices(300, MONEYNESS, MATURITES, graine=0)
    resultat = entrainer(lambda: CNN(6, 9), X, y, epoques=40, patience=40, **REGLAGES)
    # Prédire la moyenne donne une MSE normalisée proche de 1.
    assert resultat.meilleure_validation < 0.3
