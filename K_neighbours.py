from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()

# print(iris.keys())
# print(iris.DESCR)

feature = iris.data
labels = iris.target

# print(feature[0],labels[0])
clf=KNeighborsClassifier()

clf.fit(feature,labels)

pridicted=clf.predict([[31,1,1,1]])

print(pridicted)