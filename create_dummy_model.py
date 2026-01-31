from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)
joblib.dump(clf, 'student_dropout_model.joblib')
print('Dummy model saved to student_dropout_model.joblib')
