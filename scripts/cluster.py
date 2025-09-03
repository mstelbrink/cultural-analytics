import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import FEATURES, URL
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Number of clusters
k = 5

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

kmeans = KMeans(n_clusters=k, random_state=42)

df['cluster'] = kmeans.fit_predict(scaled_features)

for y in range(1970, 2020, 10):
    for x in range(0, k):
        print(str(df[(df['cluster'] == x) & (df['decade'] == y)].shape[0]) + " entries from year " + str(y) + " in cluster " + str(x))
    print("----------------------------")

mean_cluster_values = df.groupby("cluster")[FEATURES].mean()
print(mean_cluster_values)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
reduced_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
reduced_df['cluster'] = df['cluster']
reduced_df['decade'] = df['decade']

plt.figure(figsize=(12, 8))
for i in range(0, k):
    pts = reduced_df[reduced_df['cluster'] == i]
    plt.scatter(pts['PC1'], pts['PC2'], label=f"Cluster {i + 1}", alpha=0.8)

# Optional - can disturb cluster visualization when using many data points
#
# for i in range(reduced_df.shape[0]):
#     plt.annotate(reduced_df['cluster'].iloc[i], (reduced_df['PC1'].iloc[i], reduced_df['PC2'].iloc[i]), fontsize=8, alpha=0.7)

plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.grid(True)
plt.legend()
plt.show()