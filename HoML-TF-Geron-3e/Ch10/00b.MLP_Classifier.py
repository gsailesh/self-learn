from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(X_train.shape, y_train.shape)
clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50), random_state=42)
pipel = make_pipeline(StandardScaler(), clf)
pipel.fit(X_train, y_train)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
y_pred = pipel.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"{acc=}")
