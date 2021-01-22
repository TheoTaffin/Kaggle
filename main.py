from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

train_path = "dataset/train.csv"
test_path = "dataset/test.csv"


df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
corr = df_train.corr()

df_train = df_train.astype({"Sex": 'category', "Embarked": 'category'})
df_train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df_train.dropna(inplace=True)

cat_columns = df_train.select_dtypes(['category']).columns
df_train[cat_columns] = df_train[cat_columns].apply(lambda x: x.cat.codes)

features_columns = [name for name in df_train.columns if name != 'Survived']

X = df_train[features_columns]
y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = classification_report(y_test, pred)
print(score)
