import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

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

ids = []
for i in range(startYear, endYear, 10):
    results = sp.search(q='genre:' + genre + ' year:' + str(i) + '-' + str(i+10), type=type, market=market, limit=limit, offset=offset)
    for j in range(0, len(results['tracks']['items'])):
        ids.append(results['tracks']['items'][j]['id'])

years = {}
results = sp.tracks(ids)
for j in range(0, len(ids)):
    years[results['tracks'][j]['id']] = results['tracks'][j]['album']['release_date']