import keras

from keras.models import load_model
import numpy as np 

from art.attacks.evasion import ThresholdAttack
from art.classifiers import KerasClassifier
from dataset import load_carla_test_dataset

test_data = load_carla_test_dataset()
x_test = np.array([i[0] for i in test_data], dtype='float32').reshape((-1,64,64,1))
y_test = np.array([i[1] for i in test_data])

model = load_model("model.h5")

classifier = KerasClassifier(model, clip_values=(0, 255), use_logits=False)

predictions = classifier.predict(x_test)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

attack = ThresholdAttack(classifier,th=30)


x_adv = attack.generate(x_test)

predictions_adv=classifier.predict(x_adv)

accuracy = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
