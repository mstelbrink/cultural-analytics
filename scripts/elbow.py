import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import URL
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Parameters
cluster_count = 4

data = pd.read_csv(URL)

df = preprocess(data)

df['release_date'].values.astype('datetime64[Y]')
df['year'] =  pd.DatetimeIndex(df['release_date']).year
df['year'].values.astype('float64')
df['decade'] = (df['year'] // 10) * 10

df.reset_index(drop=True, inplace=True)

features = df[['danceability', 'energy', 'loudness', 'tempo', 'valence']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method to decide on the best 'K'
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()