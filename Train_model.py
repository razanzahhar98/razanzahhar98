from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


train = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation = ImageDataGenerator(rescale=1/255)
epochs = 10
test_size=0.2

train_image_gen = train.flow_from_directory('C:/Users/razan/Desktop/dataset_sunglasses/dataset',target_size=(100,100),batch_size=32,color_mode='grayscale',class_mode='binary')
valid_image_gen = validation.flow_from_directory('C:/Users/razan/Desktop/dataset_sunglasses/dataset',target_size=(100,100),batch_size=32,color_mode='grayscale',class_mode='binary')



batch_size=32
SPE= len(train_image_gen.classes)//batch_size 
VS = len(valid_image_gen.classes)//batch_size 
print(SPE,VS)

print(train_image_gen.class_indices)

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(100,100,1),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.5))


model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

results = model.fit_generator(train_image_gen,validation_data=valid_image_gen,epochs=10,steps_per_epoch=SPE,validation_steps=VS)
results.history.keys()
# accuracy=results.history['accuracy']
# val_accuracy=results.history['val_accuracy']

# plt.plot(results.history['accuracy'])
# plt.plot(results.history['val_accuracy'])
# plt.plot(accuracy, 'y', label='Traing accuracy')
# plt.plot(val_accuracy, 'b', label='Validation accuracy')
# plt.title('model accuracy')
# plt.ylabel('Training and Validation accuracy')
# plt.xlabel('epoch')
# plt.legend(["accuracy", "val_accuracy"])

# accuracy=results.history['loss']
# val_accuracy=results.history['val_loss']

# plt.plot(results.history['loss'])
# plt.plot(results.history['val_loss'])
# plt.plot(accuracy, 'y', label='Traing Loss')
# plt.plot(val_accuracy, 'b', label='Validation Loss')
# plt.title('model loss')
# plt.ylabel('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.legend(["loss", "val_loss"])

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), results.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), results.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), results.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), results.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy CNN")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")

# save plot to disk
plt.savefig('plot.png')

model.save("CNN__model.h5")