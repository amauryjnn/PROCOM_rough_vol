# Calibration profonde interprétable de modèles de volatilité rugueuse

Projet PROCOM, IMT Atlantique. Encadrant : Jean-Marc Le Caillec.

On apprend la calibration du modèle rough Heston par réseau de neurones, on
compare trois architectures (FNN / CNN / LSTM), et on utilise l'interprétabilité
comme critère de choix. Objectif final : la couverture.

Article de référence : Yuan, Brigo, Jacquier, Pede, *Deep Learning
Interpretability for Rough Volatility*, arXiv:2411.19317.

## Conventions figées — ne jamais changer sans en parler à l'équipe

**Paramétrisation rough Heston** (El Euch–Rosenbaum, celle de l'article) :

    V_t = V_0 + kappa/Gamma(H+1/2) * int_0^t (t-s)^{H-1/2} [(theta - V_s) ds + nu sqrt(V_s) dW_s]

`kappa` multiplie toute l'accolade et `nu` est à l'intérieur. Donc :
- vitesse de retour à la moyenne = `kappa`
- **vol de vol effective = `kappa * nu`**, pas `nu`

Une partie de la littérature écrit `nu` comme vol de vol directement. Si un
résultat semble décalé d'un facteur `kappa`, c'est là qu'il faut regarder.

**Ordre canonique des paramètres** : `(H, V0, theta, kappa, nu, rho)`.
C'est l'ordre des sorties du réseau et des colonnes de la jacobienne. Ne pas
le permuter.

**Grille** : 9 moneyness de 0,6 à 1,4 × 6 maturités {0,6 ; 0,9 ; 1,2 ; 1,5 ;
1,8 ; 2,0} ans = 54 points. Définie dans `config/default.yaml`, jamais en dur
dans le code. Identique pour toutes les lignes du dataset — c'est ce qui donne
un sens stable à l'entrée n du réseau.

**Découpage** : 70 / 15 / 15 (train / validation / test), pas 85 / 15.
On règle les hyperparamètres de trois architectures : si on les règle sur le
jeu de test, il n'est plus un jeu de test.

**Normalisation** : statistiques calculées sur le train UNIQUEMENT. Les
calculer sur l'ensemble fait fuiter le test dans l'entraînement.

**Graines aléatoires** : fixées et sauvegardées avec le dataset. Sans ça, on
ne peut pas savoir si un écart entre deux architectures vient du réseau ou du
jeu de données.

## Contrat de données

Une ligne du dataset :

    entrée  : 54 float64, volatilités implicites, ordre maturité externe /
              moneyness interne
    sortie  : 6 float64, (H, V0, theta, kappa, nu, rho)

Deux réseaux, pas un :
- `G_w` : surface (54) → paramètres (6). C'est l'objet de l'article, c'est lui
  qu'on compare entre architectures.
- `F_w` : paramètres (6) → surface (54). Apprend le pricer. Nécessaire parce
  qu'il donne la jacobienne `J = d sigma_imp / d Theta` par autodiff, dont les
  étapes 4 et 6 dépendent. Dériver `G_w` ne donne pas `J`.

## La chaîne et ses validations

| # | Étape | Validation |
|---|---|---|
| 1 | Dataset (surfaces rough Heston) | à `H = 1/2`, le pricer rugueux redonne Heston classique |
| 2 | Entraînement de `F_w` et `G_w` | erreur relative en carte + repricing |
| 3 | Calibration : propre → bruitée → incomplète | courbe de dégradation |
| 4 | Sensibilités `J` par autodiff | contrôle par différences finies |
| 5 | Couverture delta puis delta-vega | écart-type du P&L final |
| 6 | Interprétabilité (SHAP, LIME, DeepLIFT, LRP) | corrélation avec `J` |

Les étapes 2 à 6 sont rejouées pour chacune des trois architectures.

## Stratégie de développement

Monter toute la chaîne avec le **pricer Heston classique** d'abord : il a une
formule rapide, donc dataset, réseaux, sensibilités, couverture et
interprétabilité sont en place avant de brancher le pricer rugueux. Il ne reste
qu'à substituer une fonction. Bonus : sur cette version `H = 1/2` est connu.

Faire tourner le pipeline de bout en bout sur ~200 surfaces avant toute
génération longue.

## Pièges connus

- **Test `H = 1/2`** : gratuit, et sans lui on peut entraîner des réseaux tout
  un semestre sur des données fausses. À garder comme test automatique.
- **Quadrature de Lewis bornée**, Gauss-Legendre sur [0, u_max], pas
  Gauss-Laguerre : en rugueux le schéma explicite diverge sur les grands `u`.
- **Effet de bord du CNN** : sur une grille 9×6, 26 cases sur 54 sont sur le
  bord. Le mode de padding n'est pas un détail, il doit être testé et reporté.
  Et l'information sur `H` et `rho` est dans les ailes, donc sur le bord.
- **Un réseau peut bien approcher une fonction et mal approcher sa dérivée.**
  Un MSE minuscule ne garantit pas une bonne jacobienne. D'où le contrôle de
  l'étape 4 par différences finies.
- **Ne pas deviner les directions non identifiables en lisant les formules.**
  Les mesurer par le spectre de `J^T J`.

## Répartition

- Anthony : CNN, et les étapes 4 (sensibilités) et 6 (interprétabilité)
- Autres membres : FNN, LSTM, dataset, couverture

## Stack

Python 3.11+, NumPy / SciPy (pricers, quadrature), PyTorch (réseaux et
gradients), `shap`, PyYAML, matplotlib. Tests avec `pytest`.