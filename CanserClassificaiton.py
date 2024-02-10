import pandas as pd

dataset = pd.read_csv("breast_cancer.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                          random_state=1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train[:,:] = ss.fit_transform(x_train[:,:])
x_test[:,:] = ss.fit_transform(x_test[:,:])

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(x_train_pca,y_train)

y_pred = classifier.predict(x_test_pca)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
accs = accuracy_score(y_test, y_pred)













