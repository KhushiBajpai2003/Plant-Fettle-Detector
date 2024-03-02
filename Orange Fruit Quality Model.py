#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np


# In[2]:


IMAGE_SIZE = 255
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50
N_CLASSES = 2


# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\Project\Orange",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


import splitfolders
import os
path ="D:\Project\Orange"
print(os.listdir(path))


# In[5]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[6]:


class_names = dataset.class_names
class_names


# In[7]:


plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[8]:


len(dataset)


# In[9]:


train_size = 0.8
len(dataset)*train_size


# In[10]:


train_ds = dataset.take(63)
len(train_ds)


# In[11]:


test_ds = dataset.skip(63)
len(test_ds)


# In[12]:


val_size=0.1
len(dataset)*val_size


# In[13]:


val_ds = test_ds.take(7)
len(val_ds)


# In[14]:


test_ds = test_ds.skip(7)
len(test_ds)


# In[15]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[16]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[17]:


len(train_ds)


# In[18]:


len(val_ds)


# In[19]:


len(test_ds)


# In[20]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[21]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# In[22]:


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.4), 
])


# In[23]:


train_ds_augmented = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[24]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 2

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[25]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,  # Adjust the patience parameter
    restore_best_weights=True,
    monitor='val_loss'
)


# In[26]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[27]:


model.summary()


# In[28]:


history = model.fit(
    train_ds_augmented,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# In[33]:


scores = model.evaluate(test_ds)


# In[34]:


scores = model.evaluate(train_ds)


# In[35]:


scores


# In[36]:


tf.keras.callbacks.History()


# In[37]:


history.params


# In[38]:


history.history.keys()


# In[39]:


type(history.history['loss'])


# In[40]:


len(history.history['loss'])


# In[41]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[42]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[43]:


epochs = range(1, len(acc) + 1)  

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


# In[44]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    


# In[45]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[67]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i].numpy()]

        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[47]:


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Assuming you have a list to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Collect all unique class names from the training dataset
all_class_names = ['Good Quality', 'Orange Rotten']
label_encoder.fit(all_class_names)  

# Iterate through the test dataset to collect true and predicted labels
for images, labels in test_ds:
    for i in range(len(images)):
        try:
            true_labels.append(label_encoder.transform([class_names[labels[i].numpy()]])[0])
        except KeyError:
            # Handle unseen label gracefully
            print(f"Unseen label: {class_names[labels[i].numpy()]}")
            continue
        
        # Using your predict function
        predicted_class, _ = predict(model, images[i].numpy())
        predicted_labels.append(label_encoder.transform([predicted_class])[0])

# Convert lists to numpy arrays for precision calculation
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate precision, recall, and f1-score with zero_division parameter
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')



# In[48]:


# Calculate precision with zero_division parameter
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print(f'Precision: {precision:.2f}')


# In[49]:


# Calculate recall with zero_division parameter
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print(f'Recall: {recall:.2f}')


# In[50]:


# Calculate precision with zero_division parameter
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
print(f'F1-score: {f1:.2f}')


# In[51]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Define class names for visualization
class_names = ['Good Quality', 'Orange Rotten']

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[52]:


model.save("D:\Project\MODELS3\Orrange_Quality.h5")


# In[53]:


keras_model = tf.keras.models.load_model("D:\Project\MODELS3\Orrange_Quality.h5")


# In[54]:


converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()


# In[55]:


with open("D:\Project\MODELS\Orrange_Quality.tflite", "wb") as tflite_file:
    tflite_file.write(tflite_model)


# In[ ]:




