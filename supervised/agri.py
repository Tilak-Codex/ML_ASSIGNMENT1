import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'soil_ph': [6.5, 7.0, 5.8, 6.8, 7.2, 5.5, 6.0, 7.5],
    'moisture': [30, 45, 25, 40, 50, 20, 35, 55],
    'temperature': [25, 28, 22, 27, 30, 20, 24, 32]
}

df = pd.DataFrame(data)

# Apply KMeans
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(df)

print(df)

# Visualization
plt.scatter(df['soil_ph'], df['moisture'], c=df['cluster'])
plt.xlabel("Soil pH")
plt.ylabel("Moisture")
plt.title("Crop Pattern Clustering")
plt.show()