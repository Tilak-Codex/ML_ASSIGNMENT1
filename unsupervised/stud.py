import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: CREATE DATASET
# -------------------------------
data = {
    'study_hours': [2, 5, 1, 6, 3, 8, 2, 7, 4, 6],
    'attendance': [60, 85, 50, 90, 70, 95, 55, 88, 75, 92],
    'assignments_completed': [3, 8, 2, 9, 5, 10, 3, 9, 6, 8],
    'mobile_usage_hours': [6, 2, 7, 1, 5, 1, 8, 2, 4, 2]
}

df = pd.DataFrame(data)

# -------------------------------
# STEP 2: APPLY K-MEANS
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

print("Clustered Data:\n")
print(df)

# -------------------------------
# STEP 3: VISUALIZATION
# -------------------------------
plt.scatter(df['study_hours'], df['mobile_usage_hours'], c=df['cluster'])

plt.xlabel("Study Hours")
plt.ylabel("Mobile Usage Hours")
plt.title("Student Learning Style Clusters")

plt.show()