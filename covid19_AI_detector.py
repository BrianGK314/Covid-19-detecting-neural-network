import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC,CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE=2

training_data_generator = ImageDataGenerator(rescale=1./255,
                                             zoom_range=0.2,
                                             rotation_range = 15,
                                             width_shift_range= 0.05,
                                             height_shift_range = 0.05)

training_iterator = training_data_generator.flow_from_directory('/content/train1',
                                                                class_mode='categorical',
                                                                color_mode='grayscale',
                                                                batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator(rescale=1./255)
validation_iterator = validation_generator.flow_from_directory('/content/test1',
                                                               class_mode='categorical',
                                                               color_mode='grayscale',
                                                               batch_size=BATCH_SIZE)
model=tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256,256,1)))
model.add(tf.keras.layers.Conv2D(8,2,strides=2,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5,5),strides=(5,5)))
model.add(tf.keras.layers.Conv2D(8,2,strides=2,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(3,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[CategoricalAccuracy(),AUC()])
model.summary()

stop = EarlyStopping(monitor='categorical_accuracy',
                     mode='max',
                     verbose=1,
                     patience=3)
                     
model.fit(
    training_iterator,
    steps_per_epoch = training_iterator.samples/BATCH_SIZE,
    epochs = 15,
    validation_data=validation_iterator,
    validation_steps=validation_iterator.samples/BATCH_SIZE,
    callbacks=[stop])



to_predict = ImageDataGenerator().flow_from_directory('/content/predictcovid',class_mode='categorical',color_mode='grayscale')
predictions = model.predict(to_predict)
label_map = training_iterator.class_indices
print(label_map)
#print(predictions)
import numpy as np
import statistics
gg= np.argmax(predictions,axis=1)
print(gg)
try:
  print(statistics.mode(gg))
except Exception:
  print('No mode')
