# Projet de Cloud Computing
### 3A ENSAE 2022-2023
### Beatriz Farah et Conrad Thiounn

Dans ce projet nous avons travaillé sur l'article "Dimension Independent Matrix Square using MapReduce (DIMSUM)"  qui introduit l'algorithme DIMSUM, permettant de calculer pour une matrice creuse A le produit matriciel $A^TA$ de façon efficiente en utilisant MapReduce. Notre enjeu est de proposer une implémentation en Spark et de montrer un cas d'utilisation pratique des méthodes proposées. Dans ce rapport nous allons présenter les algorithmes et notre implémentation sur Python et Spark, aussi qu'illustrer l'utilisation des méthodes introduits dans un cas de régréssion linéaire et dans un cas de similarités cosine.

Notre répositoire est organisé de la façon suivante: 
#### Implémentation Python
Tous les notebooks utilisés sont dans le dossier ``notebooks``, qui contient:
- [Algorithms from the paper (numpy).ipynb](https://github.com/beafarah/cloudcomputingensae/blob/main/notebooks/Algorithms%20from%20the%20paper%20(numpy).ipynb) : Notebook où on implémente les algorithmes décrits dans l'article sur Numpy 
- Tests on matrix size.ipynb](https://github.com/beafarah/cloudcomputingensae/blob/main/notebooks/Tests%20on%20matrix%20size.ipynb) : Notebook où on teste les méthodes implementés sur Numpy

Les méthodes sont également trouvés dans le dossier ``src``, qui contient:
- helper.py : Fichier avec des fonctions utiles pour l'implémentation des méthodes
- methods_py.py : Fichier avec l'implémentation des l'algorithmes de l'article


#### Implémentation sur Spark
Tous les notebooks utilisés sont dans le dossier ``notebooks``, qui contient:
- [Spark_notebook_DIMSUM.ipynb](https://github.com/beafarah/cloudcomputingensae/blob/main/notebooks/Spark_notebook_DIMSUM.ipynb) : Notebook implémentant en Spark l'algorithme DIMSUM et également l'algorithme naïf en version distribué en Spark
- [Spark_notebook_DIMSUM_benchmark_version.ipynb](https://github.com/beafarah/cloudcomputingensae/blob/main/notebooks/Spark_notebook_DIMSUM_benchmark_version.ipynb) : Notebook utilisé pour le benchmark
- [Applications.ipynb](https://github.com/beafarah/cloudcomputingensae/blob/main/notebooks/Applications.ipynb) : Applications - cas d'une regression linéaire et cas des similarités cosine


