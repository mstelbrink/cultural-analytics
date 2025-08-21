import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import chunks
import math
import statsmodels.api as sm

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                                           client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")))

# Parameters
genre='pop'
type='track'
market='US'
limit = 1
offset = 0
startYear = 1970
endYear = 2020
rows = 3
columns = 4
include_subgenres = True

df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")

if include_subgenres:
    df = df[df['track_genre'].str.contains('pop')]
else:
    df = df[df['track_genre'] == 'pop']

ids = df['track_id']
chunked_ids = list(chunks(ids, 50))

dates = {}
published_in_US = {}
for i in range(0, len(chunked_ids)):
    results = sp.tracks(chunked_ids[i])
    for j in range(0, len(chunked_ids[i])):
        dates[results['tracks'][j]['id']] = results['tracks'][j]['album']['release_date']
        if market in results['tracks'][j]['album']['available_markets']:
            published_in_US[results['tracks'][j]['id']] = True
        else:
            published_in_US[results['tracks'][j]['id']] = False

df['release_date'] = df['track_id'].map(dates)
df['market_US'] = df['track_id'].map(published_in_US)

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df.dropna(subset='release_date', inplace=True)
df['numeric_release_date'] = df['release_date'].map(pd.Timestamp.toordinal)

df = df[df['release_date'].dt.year >= startYear]
df = df[df['release_date'].dt.year < endYear]
df = df[df['market_US'] == True]

df.set_index('release_date', inplace=True)

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