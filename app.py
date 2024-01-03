import tweepy
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Input, Output
from wordcloud import WordCloud
# tensorflow.keras.models import load_model  # Si vous utilisez Keras pour le modèle de prédiction de sentiments
#from tweets_analysis.wordcloud import wordcloud_
from sentiments_classifification.lexical import *
from twitter_setup.collect_tweets import collect_comments , get_tweets
from twitter_setup.twitterConnectionSetup import twitter_setup
from sentiments_classifification.predict_model import predict_polarity
#from insultes.compteur_insultes import Compteur , PatternDeFichierTexte
from wordcloud import WordCloud
import matplotlib.pyplot as plt








def wordcloud_(filtered_tweets):
    mots_cles=[w for tweet in filtered_tweets for w in tweet]
    texte = ' '.join(mots_cles)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texte)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Enregistrer le nuage de mots comme une image
    wordcloud.to_file('wordcloud_image.png')  # Enregistre l'image sous le nom "wordcloud_image.png"

    return wordcloud

    
    #plt.show()


def PatternDeFichierTexte(text) :
    #text: Path du fichier à transformer en liste de mot
    try :
        f = open(text,'r')
        lignes = f.readlines()
        for i in range(len(lignes)) :
            lignes[i]=lignes[i].replace('\n', "") #enlève les retours à la ligne de la liste lignes afin d'avoir bien une insulte en string
        return lignes
    except IOError as e:
        print("error occured while reading the file ", e)



def Compteur(Listetweet, patternL):
    #patternL=Liste d'insulte
    #Lisetweet: liste de tweet, chaque élément est un string contenant un tweet
    a=0 #Compteur
    set={} #Dictionnaire avec en clé l'insulte utilité et en valeur son occurence
    for i in range(len(Listetweet)):
        Ltweet=Listetweet[i].split() #liste des mots du tweet en question
        for j in range(len(Ltweet)) :
            Ltweet[j].replace(',', " ") #enlève les virgules afin d'éviter les répétitions dans le dictionnaire
            Ltweet[j].replace("-"," ") #de même
            if Ltweet[j] in patternL:
                a+=1
                if Ltweet[j] in set :
                    set[Ltweet[j]]+=1
                else :
                    set[Ltweet[j]]=1
    return a, set



insultes = PatternDeFichierTexte(r'C:\Users\HOME\twitty_gate\insultes\insultes.txt')
tweets = ''
taux = 0










# Charger le modèle de prédiction de sentiments

#model = load_model(r'C:\Users\HOME\twitty_gate\sentiments_classifification\trained_model.joblib')



# Initialise l'application Dash
app = Dash(__name__)

# Layout de l'application Dash
div_styles = {
    'textAlign': 'center',
    'margin': '20px',
}

input_styles = {
    'marginRight': '10px',
}

button_styles = {
    'backgroundColor': 'blue',
    'color': 'white',
    'borderRadius': '5px',
}

output_wordcloud_styles = {
    'marginTop': '30px',
    'textAlign': 'center',
}

output_sentiment_styles = {
    'marginTop': '30px',
}

output_amability_styles = {
    'marginTop': '30px',
    'textAlign': 'center',
}

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Twitter Analysis Dashboard", style={'textAlign': 'center', 'color': 'blue'}),

    html.Div([
        dcc.Input(id='username', type='text', placeholder='Enter Twitter Username', style=input_styles),
        dcc.Input(id='num_tweets', type='number', placeholder='Number of Tweets', style=input_styles),
        html.Button('Analyze', id='submit-val', n_clicks=0, style=button_styles)
    ], style=div_styles),

    html.Div(id='output-wordcloud', style=output_wordcloud_styles),
    dcc.Graph(id='output-sentiment', style=output_sentiment_styles),
    html.Div(id='output-amability', style=output_amability_styles)
])
# Callback pour générer le WordCloud
@app.callback(
    Output('output-wordcloud', 'children'),
    Input('submit-val', 'n_clicks'),
    [Input('username', 'value'), Input('num_tweets', 'value')]
)




#def update(n_clicks , username , num_tweets):
    #tweets = get_tweets(username, num_tweets)



def update_wordcloud(n_clicks, username, num_tweets):
    if n_clicks > 0:

        

        wordcloud = wordcloud_(listemot(remove_stopwords(tweets)))
        return html.Img(src=wordcloud.to_image())

# Callback pour générer l'histogramme de sentiments
@app.callback(
    Output('output-sentiment', 'figure'),
    Input('submit-val', 'n_clicks'),
    [Input('username', 'value'), Input('num_tweets', 'value')]
)
def update_sentiment_histogram(n_clicks, username, num_tweets):
    if n_clicks > 0:
        tweets = get_tweets(username, num_tweets)
        sentiments = [predict_polarity(tweet) for tweet in tweets]
        taux = len([x for x in sentiments if x==0])/len([x for x in sentiments if x==1])
        df = pd.DataFrame({'Sentiment': sentiments})
        fig = px.histogram(df, x='Sentiment')
        return fig

# Callback pour afficher l'indice d'amabilité
@app.callback(
    Output('output-amabilite', 'children'),
    Input('submit-val', 'n_clicks'),
    [Input('username', 'value'), Input('num_tweets', 'value')]
)
def update_amabilite(n_clicks, username, num_tweets):
    if n_clicks > 0:

        a = Compteur(tweets , insultes)[0]

        avg_amabilite = taux / a

        return html.Div(f'Indice d\'amabilité moyen: {avg_amabilite}')
    

@app.callback(
    Output('profile-image', 'src'),
    Input('submit-profile', 'n_clicks'),
    [Input('username', 'value')]
)
def update_profile_image(n_clicks, username):
    if n_clicks > 0:
        try:
            api = twitter_setup()
            user = api.get_user(username)
            profile_image_url = user.profile_image_url_https  # Fetch profile image URL
            return profile_image_url  # Set the src attribute of the Img component to display the image
        except tweepy.TweepError as e:
            print(f"Error: {e}")
            return None



# Exécute l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
