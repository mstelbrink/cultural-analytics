import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import FEATURES, URL
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

features = df[FEATURES]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method to decide on the best number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Trägheit')
plt.show()