import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')
validation_set = np.load('validation_set.npy')

# pca = PCA(n_components=7)  # Número desejado de componentes principais
# train_feat_pca = pca.fit_transform(train_set[:,:-1])
# test_feat_pca = pca.transform(test_set[:,:-1])
# validation_feat_pca = pca.transform(validation_set[:,:-1])

train_feat = train_set[:,:-1]
train_label = train_set[:,-1]
test_feat = test_set[:,:-1]
test_label = test_set[:,-1]
validation_feat = validation_set[:,:-1]
validation_label = validation_set[:,-1]

Accur = []
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_feat, train_label)
    pred = knn.predict(validation_feat)
    acc = accuracy_score(validation_label, pred)
    Accur.append(acc * 100)

x = list(range(1,11))
plt.figure()
plt.plot(x, Accur)
plt.xlabel('K')
plt.ylabel('Acurácia')
plt.title('KNN')

print("Melhor Acurácia: " + str(max(Accur)))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_feat, train_label)
pred = knn.predict(test_feat)
conf_arr = np.zeros((3,3))
for i in range(len(pred)):
    conf_arr[int(pred[i])][int(test_label[i])] += 1

print("Confusion Matrix:")
print(conf_arr)

plt.show()

