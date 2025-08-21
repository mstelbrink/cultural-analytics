import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from utils import chunks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

df = df[df['track_genre'] == 'pop']

ids = []
for i in range(0, len(df.index)):
    ids.append(df.iloc[i]['track_id'])

dates = {}

chunked_ids = list(chunks(ids, 50))

for i in range(0, len(chunked_ids)):
    results = sp.tracks(chunked_ids[i])
    for j in range(0, len(chunked_ids[i])):
        dates[results['tracks'][j]['id']] = results['tracks'][j]['album']['release_date']

df['release_date'] = df['track_id'].map(dates)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df.dropna(subset='release_date', inplace=True)
df['release_date'].values.astype('datetime64[Y]')
df['year'] =  pd.DatetimeIndex(df['release_date']).year
df['year'].values.astype('float64')
df['decade'] = (df['year'] // 10) * 10

df.reset_index(drop=True, inplace=True)

features = df[['danceability', 'energy', 'loudness', 'tempo', 'valence']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)

df['cluster'] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
reduced_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
reduced_df['cluster'] = df['cluster']
reduced_df['decade'] = df['decade']

plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_df['PC1'], reduced_df['PC2'], c=reduced_df['cluster'], cmap='viridis', alpha=0.6)

for i in range(reduced_df.shape[0]):
    plt.annotate(reduced_df['decade'].iloc[i], (reduced_df['PC1'].iloc[i], reduced_df['PC2'].iloc[i]), fontsize=8, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Songs with Decade Annotations (PCA Reduced)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()