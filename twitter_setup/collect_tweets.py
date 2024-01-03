from twitter_setup.twitterConnectionSetup import *

'''def collect_by_user_liste(user_id,n):
    #Renvoie une liste de tweet pour un utilisateur donné
    l=[]
    connexion = twitter_setup()
    statuses = connexion.user_timeline(id = user_id, count = n)
    for status in statuses:
        l.append(status.text) #chaque élément de la liste est un tweet de l'utilisateur
    return l'''

'''def collect_by_user_unseulstring(user_id,n):
    l = collect_by_user_liste(user_id,n)
    rep = ""
    for x in l:
        rep += x
    return rep'''

def get_tweets(username, num_tweets):

    api = twitter_setup()

    # Récupérer les tweets de l'utilisateur
    tweets = api.user_timeline(screen_name=username, count=num_tweets, tweet_mode="extended")
    return [tweet.full_text for tweet in tweets]


def collect_comments(user_id ):

    connexion = twitter_setup()

    statuses = connexion.user_timeline(id = user_id, count = 10)

    comments_seq=[]

    for status in statuses:

        tweet_id = status.tweet_id

        comments = tweepy.Cursor(connexion.search, q=f"to:{connexion.get_status(tweet_id).user.screen_name}",
                            since_id=tweet_id, tweet_mode='extended').items()

        # Print comments (limited to a certain number of comments for example purposes)

        
        comment_count = 0
        for comment in comments:
            #print(f"Comment by @{comment.user.screen_name}: {comment.full_text}")
            comments_seq.append(comment.full_text)

            comment_count += 1
            if comment_count >= 10:  # Adjust the number of comments you want to retrieve
                break

    return comments_seq



#print(collect_by_user_liste('EmmanuelMacron'))