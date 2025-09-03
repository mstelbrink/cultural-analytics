import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import URL, FEATURES
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

# Print variance for each feature
for x in range(0, len(FEATURES)):
    print(FEATURES[x] + ": " + str(df[FEATURES[x]].var(ddof=0)))

# Create seasonal component
yearly_audio_features = df[FEATURES].resample('YE').mean()

fig, axs = plt.subplots(rows, columns)
fig.subplots_adjust(hspace=0.33)

for i in range(0, len(FEATURES)):
    a = df['numeric_release_date']
    b = df[FEATURES[i]]
    a = sm.add_constant(a)

    model = sm.OLS(b, a).fit()
    predictions = model.predict(a)

    x = math.floor(i / columns)
    y = i % columns

    axs[x, y].scatter(df.index, df[FEATURES[i]], alpha=0.1)
    axs[x, y].boxplot(df[FEATURES[i]])
    axs[x, y].plot(df.index, predictions, color='purple')
    axs[x, y].plot(yearly_audio_features.index, yearly_audio_features[FEATURES[i]], color='red', alpha=0.8)
    axs[x, y].set_title(FEATURES[i])
    axs[x, y].grid()

plt.show()