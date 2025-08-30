import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from utils.preprocessor import preprocess
from utils.constants import URL, FEATURES_TO_PLOT
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

yearly_audio_features = df[FEATURES_TO_PLOT].resample('YE').mean()

fig, axs = plt.subplots(rows, columns)
fig.subplots_adjust(hspace=0.33)

for i in range(0, len(FEATURES_TO_PLOT)):
    a = df['numeric_release_date']
    b = df[FEATURES_TO_PLOT[i]]
    a = sm.add_constant(a)

    model = sm.OLS(b, a).fit()
    predictions = model.predict(a)

    x = math.floor(i / columns)
    y = i % columns

    axs[x, y].scatter(df.index, df[FEATURES_TO_PLOT[i]], alpha=0.1)
    axs[x, y].plot(df.index, predictions, color='yellow')
    axs[x, y].plot(yearly_audio_features.index, yearly_audio_features[FEATURES_TO_PLOT[i]], color='red', alpha=0.8)
    axs[x, y].set_title(FEATURES_TO_PLOT[i])
    axs[x, y].grid()

plt.show()