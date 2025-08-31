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

kmeans = KMeans(n_clusters=cluster_count, random_state=42)

df['cluster'] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
reduced_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
reduced_df['cluster'] = df['cluster']
reduced_df['decade'] = df['decade']

plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_df['PC1'], reduced_df['PC2'], c=reduced_df['cluster'], cmap='viridis', alpha=0.6)

# Optional - can disturb cluster visualization when using many data points
#
# for i in range(reduced_df.shape[0]):
#     plt.annotate(reduced_df['decade'].iloc[i], (reduced_df['PC1'].iloc[i], reduced_df['PC2'].iloc[i]), fontsize=8, alpha=0.7)

plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.title('K-Means Cluster')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()