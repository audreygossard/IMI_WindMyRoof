## Projet Wind My Roof

#### Pistes

Lorsque l'on fait varier C_e on observe que omega ne varie pas directement (inertie).
Cela empêche d'appliquer une méthode de gradient directement puisque C_e a alors tendance à s'emballer.

Notre idée consiste à séparer les itérations de gradient d'un temps caractéristique de retour à l'équilibre pour omega.
Nous allons ainsi chercher à identifier ce temps caractéristique et étudier sa dépendance aux différents paramètres du problème, à la variation de C_e imposée et à la valeur initiale de omega.

Une seconde idée consiste à d'abord imposer un delta C_e plus grand que celui visé pour réduire le temps de retour à l'équilibre et ensuite remettre C_e à la valeur cible : on obtiendrait alors une itération plus rapide.

### Pour le 15 mars

- Identifier les paramètres qui ont une influence sur le temps de relaxation
- Fonction qui renvoit le temps de relaxation pour différentes valeurs de paramètres (tableau de dimension le nbr de paramètres qui influent sur le temps de relaxation)
- Fonction d'interpolation qui prend en entrée des paramètres et renvoit le temps de relaxation.