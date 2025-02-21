import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier,GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import pickle

#  Load Dataset
df = pd.read_csv("creditcard.csv", skiprows=lambda i: i > 0 and np.random.rand() > 0.5)
print("Dataset Loaded")

#  Handling Missing Values
num_imputer = SimpleImputer(strategy='median')
df.iloc[:, :-1] = num_imputer.fit_transform(df.iloc[:, :-1])

#  Outlier Detection & Removal (Z-score)
z_scores = np.abs(zscore(df.drop(columns=['Class'])))
df = df[(z_scores < 3).all(axis=1)]
print("Outliers Removed")

#  Remove Duplicates
df.drop_duplicates(inplace=True)

#  Feature Engineering: Log Transformation on Amount
df['log_amount'] = np.log1p(df['Amount'])
df.drop(['Amount'], axis=1, inplace=True)

#  Feature Scaling (Standardization & MinMax Scaling)
scaler = StandardScaler()
df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])

#  Dimensionality Reduction using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(df.drop(columns=['Class']))


# Save the scaler and PCA transformer
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("pca_transformer.pkl", "wb") as pca_file:
    pickle.dump(pca, pca_file)

print("PCA and Scaler saved successfully!")

#  Splitting Data into Train & Test
X = X_pca
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handling Class Imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

 # Ensemble Model Selection
rf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
xgb = XGBClassifier(objective='binary:logistic', n_jobs=1, random_state=42, n_estimators=50)
stacking_classifier = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=GradientBoostingClassifier(n_estimators=50),
    n_jobs=1
)

#  Cross-Validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking_classifier, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=1)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

#  Train Final Model
stacking_classifier.fit(X_train, y_train)

with open("fraud_detection_model.pkl", "wb") as model_file:
    pickle.dump(stacking_classifier, model_file)

print(" File Saved")

y_pred = stacking_classifier.predict(X_test)

#  Evaluation Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#  ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


