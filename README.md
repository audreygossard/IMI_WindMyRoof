### Wind My Roof

##### Pistes

Lorsque l'on fait varier C_e on observe que omega ne varie pas directement (inertie).
Cela empêche d'appliquer une méthode de gradient directement puisque C_e a alors tendance à s'emballer.

Notre idée consiste à séparer les itérations de gradient d'un temps caractéristique de retour à l'équilibre.
Nous allons ainsi chercher à identifier ce temps caractéristique et étudier sa dépendance aux différents paramètres du problème, à la variation de C_e imposée et à la valeur initiale de omega.

Une seconde idée consiste à d'abord imposer un delta C_e plus grand que celui visé pour réduire le temps de retour à l'équilibre et ensuite remettre C_e à la valeur cible : on obtiendrait alors une itération plus rapide.

