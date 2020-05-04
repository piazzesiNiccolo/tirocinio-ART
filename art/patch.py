import numpy as np 
from art.classifiers import KerasClassifier
from art.attacks import AdversarialPatch
from dataset import load_carla_test_dataset
from keras.models import load_model

test_data = load_carla_test_dataset()
x_test = np.array([i[0] for i in test_data], dtype='float32').reshape((-1,64,64,1))
y_test = np.array([i[1] for i in test_data])

model = load_model('model.h5')
classifier = KerasClassifier(model=model, clip_values=(0, 255))
model.summary()
predictions = classifier.predict(x_test)
acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on bening test examples: {}%".format(accuracy * 100))
attack = AdversarialPatch(classifier)
adv = attack.generate(x_test,y_test)

predictions = classifier.predict(adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))