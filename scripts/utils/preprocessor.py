import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd

def preprocess(df):

    load_dotenv()

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                                           client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")))

    # Parameters
    genre='pop'
    market='US'
    startYear = 1970
    endYear = 2020
    include_subgenres = True

    if include_subgenres:
        df = df[df['track_genre'].str.contains(genre)]
    else:
        df = df[df['track_genre'] == genre]

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
    df['release_date'] = df.index

    return df

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]