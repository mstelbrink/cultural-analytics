import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import URL
import math
import statsmodels.api as sm

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                                           client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")))

# Parameters
rows = 3
columns = 3

data = pd.read_csv(URL)

df = preprocess(data)

features_to_plot = ['acousticness',
                    'danceability',
                    'energy',
                    'instrumentalness',
                    'liveness',
                    'loudness',
                    'mode',
                    'tempo',
                    'valence']

fig, axs = plt.subplots(rows, columns)
fig.subplots_adjust(hspace=0.33)

for i in range(0, len(features_to_plot)):
    a = df['numeric_release_date']
    b = df[features_to_plot[i]]
    a = sm.add_constant(a)

    model = sm.OLS(b, a).fit()
    predictions = model.predict(a)

    x = math.floor(i / columns)
    y = i % columns

    axs[x, y].scatter(df.index, df[features_to_plot[i]])
    axs[x, y].plot(df.index, predictions, color='red')
    axs[x, y].set_title(features_to_plot[i])
    axs[x, y].grid()

plt.show()