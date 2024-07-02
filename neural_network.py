import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')
validation_set = np.load('validation_set.npy')

# Preprocess the data
train_label = train_set[:,-1] * 2
train_feat = train_set[:,2:-1]

validation_label = validation_set[:,-1] * 2
validation_feat = validation_set[:,2:-1]

test_label = test_set[:,-1] * 2
test_feat = test_set[:,2:-1]

# Convert the labels to one-hot encoding
enc = OneHotEncoder(categories='auto')
train_label = enc.fit_transform(train_label.reshape(-1, 1)).toarray()
validation_label = enc.transform(validation_label.reshape(-1, 1)).toarray()
test_label = enc.transform(test_label.reshape(-1, 1)).toarray()

# Train the neural network
clf = MLPClassifier(hidden_layer_sizes=(30, 10), max_iter=40, batch_size=20, learning_rate_init=0.001, momentum=0.1, random_state=0)
clf.fit(train_feat, train_label)

# Make predictions
train_pred = clf.predict(train_feat)
validation_pred = clf.predict(validation_feat)
test_pred = clf.predict(test_feat)

# Calculate accuracy
train_accuracy = accuracy_score(train_label, train_pred)
validation_accuracy = accuracy_score(validation_label, validation_pred)
test_accuracy = accuracy_score(test_label, test_pred)

print("Train accuracy: {:.2f}".format(train_accuracy))
print("Validation accuracy: {:.2f}".format(validation_accuracy))
print("Test accuracy: {:.2f}".format(test_accuracy))

# Calculate and plot confusion matrix for the test results
conf_arr = confusion_matrix(np.argmax(test_label, axis=1), np.argmax(test_pred, axis=1))
label = ["Galaxy", "Star", "Quasar"]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_arr, interpolation='nearest', cmap='PiYG')
fig.colorbar(cax)
ax.set_xticklabels(['']+label)
ax.set_yticklabels(['']+label)
ax.set_title('Labels')
ax.set_ylabel('Outputs')

for (x, y), value in np.ndenumerate(conf_arr):
    plt.text(x, y, int(value), va="center", ha="center")

plt.show()
