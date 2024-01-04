# TWITTY_GATE

Il s'agit d'une application permettant l'analyse du profil d'un utilisateur twitter dans le but de déterminer sa 'toxicité'  en se basant sur l'utilisation d'insultes dans ses tweets et leur polarité ainsi que l'impression des gens sur ses tweets. Cette application est normalement destinée au grand public qui peut l'utiliser mais son principe peut être exploité par des algorithmes de régulation Twitter afin de détecter les utilisateurs les plus toxiques sur la plateforme.

## Usage

Vous pouvez récupérer l'image du projet à partir de Dockerhub


```bash

docker pull ilyass7m/twitty_gate
docker run -p 8080:80 ilyass7m/twitty_gate:latest

```

Comme vous pouvez cloner ce dépôt git en faisant attention d'installer les modules et leurs versions correspondantes

```bash
pip install  requirements.txt

```



## Structure du projet

### MVP0

Le module **twitter_setup** permet la configuration de  l'API twitter et définit des fonctionalités de collecte des tweets d'un utilisateur donné ainsi que les commentaires sur un tweet donné

### MVP1

- Implémentation de fonctions de preprocessing du texte avec [nltk](https://www.nltk.org/) et affichage du wordcloud des mots clés les plus utilisés.
- Compteur du nombre d'insultes dans un ensemble de tweets.


![image](https://github.com/ilyass7m/Twitty_Gate/assets/142548463/dc0c53e8-df9c-4ae4-b5a8-a8c96c68031e)


### MVP2

#### Polarité des tweets:

Elaboration d'un modèle de réseaux de neuronnes pour la classification des tweets en fonctions de leurs polarités . Après un travail de traitement du dataset dans l'optique du traitement de langauage naturel en utilisant les méthodes de preprocessing (tokenization , padding ...) , j'ai déployé un modèle qui repose sur le [word Embedding](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp) dont la structure est la suivante :

![image](https://github.com/ilyass7m/Twitty_Gate/assets/142548463/190d03b7-ce9c-4f9a-82a2-5593bff65e2d)

En se basant sur la polarité des tweets de l'utilisateur et sur le nombres d'insultes détectées , on a défini un 'indice d'amabilité' de la personne . 


### Réctions et impressions de l'audience 

Pour le traitement des commentaires relatifs au twitter de l'utilisateur , j'ai opté pour un modèle SVM , en vue de sa rapidité par rapport au réseau de neuronnes ci-haut .







![tempsnip](https://github.com/ilyass7m/Twitty_Gate/assets/142548463/f424c237-1698-44f3-a7ce-e90f5dd918ef)



### MVP3

Le fichier **app.py** constitue le fichier principal du projet , il rassemble toutes les fonctionnalitées et services décrits dans une application [Dash](https://dash.plotly.com/tutorial)

## Présentation du Projet 

https://docs.google.com/presentation/d/1J99qDruBkzWQZjGDGByZhQQs2v9uSVboohd_1nd-s-Y/edit#slide=id.p1




## Note

Vous remarquerez certainement lorsque vous exécuterez l'application du projet qu'elle est non fonctionnelle , ceci est principalement du au changement de la politique de twitter désormais X en ce qui concerne l'accès aux api . Bref , c'est payant et c'est plus restreignant . (https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api)

