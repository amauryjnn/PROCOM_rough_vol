import pytest
import torch

from roughvol.models.cnn import CNN

MODES = ["zeros", "replicate", "reflect"]


@pytest.mark.parametrize("padding_mode", MODES)
def test_forme_de_sortie(padding_mode):
    modele = CNN(6, 9, padding_mode=padding_mode)
    sortie = modele(torch.zeros(4, 54))
    assert sortie.shape == (4, 6)


@pytest.mark.parametrize("padding_mode", MODES)
def test_le_gradient_atteint_tous_les_poids(padding_mode):
    modele = CNN(6, 9, padding_mode=padding_mode)
    perte = modele(torch.randn(4, 54)).pow(2).mean()
    perte.backward()
    for nom, poids in modele.named_parameters():
        assert poids.grad is not None, nom
        assert poids.grad.abs().sum() > 0, nom
