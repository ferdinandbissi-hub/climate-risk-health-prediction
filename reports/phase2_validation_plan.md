# Phase 2 — Plan de Validation et Hypothèses de Features

## 1. Synthèse de l'Analyse Exploratoire (EDA)
* **Variables climatiques** : CHIRPS (précipitations), ERA5 (températures/humidité), NDVI (végétation).
* **Distribution spatiale** : 39 localités réparties sur différentes zones géographiques et altitudes.
* **Aspect temporel** : Données étalées de 2007 à 2022.

## 2. Hypothèses d'Ingénierie de Variables (Feature Engineering)
1. **Effets retardés (Lags temporels)** : Les impacts sanitaires des événements climatiques (ex. fortes pluies, vagues de chaleur) apparaissent souvent avec un décalage de 7, 14 ou 30 jours.
2. **Cumuls et Moyennes glissantes** : Calculer les précipitations cumulées sur 7, 14 et 30 jours pour capturer le risque d'inondation ou de prolifération vectorielle.
3. **Anomalies climatiques** : Écarts par rapport à la moyenne saisonnière historique locale.

## 3. Stratégie de Validation Croisée (Cross-Validation)
* **Schéma retenu** : Stratified Group K-Fold (5 folds).
* **Groupes** : Basés sur la variable `location` pour éviter toute fuite d'information spatiale (*spatial data leakage*).
* **Stratification** : Assurer une proportion constante de la variable cible `is_climate_sensitive` dans chaque fold.
