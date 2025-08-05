import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from utils import chunks

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                                           client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")))

genre='pop'
type='track'
market='US'
limit = 1
offset = 0
startYear = 1970
endYear = 2020

df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")

filtered_df = df[df['track_genre'] == 'pop']

ids = []
for i in range(0, len(filtered_df.index)):
    ids.append(filtered_df.iloc[i]['track_id'])

years = {}

chunked_ids = list(chunks(ids, 50))

for i in range(0, len(chunked_ids)):
    results = sp.tracks(chunked_ids[i])
    for j in range(0, len(chunked_ids[i])):
        years[results['tracks'][j]['id']] = results['tracks'][j]['album']['release_date']

filtered_df['release_date'] = filtered_df['track_id'].map(years)

sorted_and_filtered_df = filtered_df.sort_values(by="release_date")

plt.scatter(sorted_and_filtered_df['release_date'], sorted_and_filtered_df['danceability'])

plt.show()