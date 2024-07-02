import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')
validation_set = np.load('validation_set.npy')

# pca = PCA(n_components=7)  # Número desejado de componentes principais
# train_feat_pca = pca.fit_transform(train_set[:,:-1])
# test_feat_pca = pca.transform(test_set[:,:-1])
# validation_feat_pca = pca.transform(validation_set[:,:-1])

def decision_tree(max_feat, num_tree, train_set=train_set, validation_set=validation_set): 
    acc_list = []
    conf_arr = np.zeros((3,3))
    for i in range(1):
        forest = RandomForestClassifier(n_estimators=num_tree, max_features=max_feat, random_state=0)
        forest.fit(train_set[:,:-1], train_set[:,-1])
        pred = forest.predict(validation_set[:,:-1])

        acc = accuracy_score(validation_set[:,-1], pred)
        acc_list.append(acc * 100)

        conf_arr += confusion_matrix(validation_set[:,-1], pred, labels=[0, 0.5, 1])

    average_accuracy = sum(acc_list) / len(acc_list)
    print("Max feature: {} Number of tree: {} Accuracy value: {:.2f}".format(max_feat, num_tree, average_accuracy))
    return conf_arr

def plot_conf(conf_arr):
    label = ["Galáxia", "Estrela", "Quasar"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_arr, interpolation='nearest', cmap='PiYG')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+label)
    ax.set_yticklabels(['']+label)
    ax.set_title('Objetos Estelares')
    ax.set_ylabel('Resultados')
    
    for (x, y), value in np.ndenumerate(conf_arr):
        plt.text(x, y, int(value), va="center", ha="center")

conf_arr = decision_tree(max_feat=9, num_tree=80)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=8, num_tree=80)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=7, num_tree=80)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=9, num_tree=40)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=8, num_tree=40)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=7, num_tree=40)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=9, num_tree=20)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=8, num_tree=20)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=7, num_tree=20)
plot_conf(conf_arr)    

print()
conf_arr = decision_tree(max_feat=9, num_tree=80, train_set=train_set, validation_set=test_set)
plot_conf(conf_arr)
