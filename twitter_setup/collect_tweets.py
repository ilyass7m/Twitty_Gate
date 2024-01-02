from twitterConnectionSetup import *

def collect_by_user_liste(user_id,n):
    #Renvoie une liste de tweet pour un utilisateur donné
    l=[]
    connexion = twitter_setup()
    statuses = connexion.user_timeline(id = user_id, count = n)
    for status in statuses:
        l.append(status.text) #chaque élément de la liste est un tweet de l'utilisateur
    return l

def collect_by_user_unseulstring(user_id,n):
    l = collect_by_user_liste(user_id,n)
    rep = ""
    for x in l:
        rep += x
    return rep