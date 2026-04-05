import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'temperature': [30, 25, 28, 35, 22, 27, 33, 26],
    'humidity': [80, 60, 75, 90, 55, 70, 85, 65],
    'rainfall': [200, 150, 180, 220, 140, 160, 210, 155],
    'disease': [1, 0, 1, 1, 0, 0, 1, 0]  # 1 = Diseased, 0 = Healthy
}

df = pd.DataFrame(data)

X = df[['temperature', 'humidity', 'rainfall']]
y = df['disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Test new data
new_data = [[29, 78, 190]]
prediction = model.predict(new_data)

print("Predicted (1=Disease, 0=Healthy):", prediction)