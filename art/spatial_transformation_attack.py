import numpy as np
import keras
from art.attacks.evasion import SpatialTransformation
from art.classifiers import KerasClassifier
from dataset import load_carla_test_dataset
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D,MaxPool2D, Dropout, Flatten, Dense
from keras.models import load_model

# Step 1: load dataset

test_data = load_carla_test_dataset()
x_test = np.array([i[0] for i in test_data], dtype='float32').reshape((-1,64,64,1))
y_test = np.array([i[1] for i in test_data])

# Step 2: Create the ART classifier
model = load_model('model.h5')
classifier = KerasClassifier(model=model, clip_values=(0, 255), use_logits=False)

# Step 3: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


# Step 4: Generate adversarial test examples

attack = SpatialTransformation(classifier=classifier,
                                max_translation=70,
                                num_translations=3,
                                max_rotation=110,
                                num_rotations=2)

x_test_adv = attack.generate(x=x_test)

# Step 5: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))




