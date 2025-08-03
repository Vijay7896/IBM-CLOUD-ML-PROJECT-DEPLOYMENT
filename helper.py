import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

df_2=pd.read_csv('PMGSY_DATASET.csv')
df_2=df_2.iloc[:,:-1]
df_2.dropna(inplace=True)
target_column = 'PMGSY_SCHEME'

unique_states = df_2['STATE_NAME'].unique()
unique_districts = df_2['DISTRICT_NAME'].unique()

print("Unique STATE_NAME values:")
print(unique_states)

print("\nUnique DISTRICT_NAME values:")
print(unique_districts)

y = df_2[target_column]
# Label encode all categorical columns
label_encoders = {}
for col in df_2.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    df_2[col] = le.fit_transform(df_2[col])
    label_encoders[col] = le

X = df_2.drop(columns=[target_column])
joblib.dump(label_encoders, 'label_encoders.pkl')

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler, 'minmax_scaler.pkl')

# Train/test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Data preprocessing complete.")
print("X_train shape:", X1_train.shape)
print("X_test shape:", X1_test.shape)

target_le = LabelEncoder()
y1_train_enc = target_le.fit_transform(y1_train)
y1_test_enc = target_le.transform(y1_test)

model=RandomForestClassifier(random_state=42)
feature_names = X1_train.columns.tolist()
class_names = target_le.classes_

model.fit(X1_train, y1_train_enc)
joblib.dump(model, 'random_forest_model.pkl')

y1_pred = model.predict(X1_test)
acc = accuracy_score(y1_test_enc, y1_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y1_test_enc, y1_pred, target_names=class_names))
