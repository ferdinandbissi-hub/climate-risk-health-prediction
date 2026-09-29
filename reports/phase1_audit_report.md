# Rapport d'Audit & Synthèse Phase 1 — Doriane

## 1. Résumé du Dictionnaire des Données
* **Total d'enregistrements** : 4 176 dans `climate_features.csv`, 3 146 dans `Train.csv` (1 030 pour le jeu de test).
* **Target (`is_climate_sensitive`)** : Binaire (65,07 % de cas positifs $1$, 34,93 % de cas négatifs $0$).
* **Variables démographiques/spatiales** : `age`, `gender`, `zone`, `latitude`, `longitude`, `location` (39 localités).
* **Variables climatiques** : Précipitations (CHIRPS Daily), Températures (ERA5 Reanalysis), Végétation (NDVI), Altitude/Pente (SRTM).

## 2. Audit de Cohérence Géo-Temporelle
* **Unicité des IDs** : 0 doublon. 100% des IDs de `Train.csv` sont présents dans `climate_features.csv`.
* **Couverture temporelle** : Dates de décès du 02/10/2007 au 13/12/2022 avec correspondance exacte à 100 %.
* **Qualité** : 0 valeur manquante observée sur les variables initiales.
