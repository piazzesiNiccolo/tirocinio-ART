
from art.classifiers import KerasClassifier
from art.attacks.evasion import SpatialTransformation
import keras
import numpy as np 
from dataset import load_carla_train_dataset, load_carla_test_dataset
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,InputLayer, MaxPool2D, Dropout
from keras.optimizers import Adam

train_data = load_carla_train_dataset()
x_train = np.array([i[0] for i in train_data],dtype="float32").reshape((-1,64,64,1))
y_train = np.array([i[1] for i in train_data])

test_data = load_carla_test_dataset()
x_test = np.array([i[0] for i in test_data], dtype='float32').reshape((-1,64,64,1))
y_test = np.array([i[1] for i in test_data])




model = Sequential()

model.add(Conv2D(filters=20, kernel_size=5,strides=1, padding='same', activation='relu', input_shape=[64,64,1]))

model.add(Conv2D(filters=32, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))


model.add(Conv2D(filters=50, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))


model.add(Conv2D(filters=80, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))


model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.01), metrics=["accuracy"]
)
model.fit(x_train,y_train,batch_size=64,epochs=10)
model.summary()
model.save("model.h5")
#model = load_model('model.h5')
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




