#!/usr/bin/env python
# coding: utf-8

# **Import all the Dependencies**

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np


# **Set all the Constants**

# In[12]:


IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50
N_CLASSES = 3


# **Import data into tensorflow dataset object**

# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
   "D:\Project\Processed Images_Fruits",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names = dataset.class_names
class_names


# In[5]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# **Visualize some of the images from our dataset**

# In[6]:


# Create a mapping between numerical labels and class names
class_names = {0: 'Bad Quality_Fruits', 1: 'Good Quality_Fruits', 2: 'Mixed Quality_Fruits'}

unique_labels = set()

for _, labels_batch in dataset.take(1):
    unique_labels.update(labels_batch.numpy())

print("Unique Labels/Classes in the Dataset:", unique_labels)

# Convert numerical labels to class names for visualization
class_names_batch = [class_names[label] for label in labels_batch.numpy()]


# In[10]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):  # Display examples from 3 batches
    labels_batch_np = labels_batch.numpy()  # Convert the entire labels_batch tensor to a NumPy array
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)  # 3 batches * 4 subplots per batch
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        
        # Use the NumPy array as an index in the class_names dictionary
        label_key = labels_batch_np[i]
        plt.title(class_names[label_key])
        
        plt.axis("off")

plt.show()


# **Function to Split Dataset**
# 
# Dataset should be bifurcated into 3 subsets, namely:
# 
# 1.Training: Dataset to be used while training
# 
# 2.Validation: Dataset to be tested against while training
# 
# 3.Test: Dataset to be tested against after we trained a model

# In[7]:


len(dataset)


# In[8]:


train_size = 0.8
len(dataset)*train_size


# In[9]:


train_ds = dataset.take(60)
len(train_ds)


# In[10]:


test_ds = dataset.skip(60)
len(test_ds)


# In[11]:


val_size=0.1
len(dataset)*val_size


# In[12]:


val_ds = test_ds.take(7)
len(val_ds)


# In[13]:


test_ds = test_ds.skip(7)
len(test_ds)


# In[14]:


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


# In[15]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[16]:


len(train_ds)


# In[17]:


len(val_ds)


# In[18]:


len(test_ds)


# **Cache, Shuffle, and Prefetch the Dataset**

# In[19]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# **Building the Model**
# 
# **Creating a Layer for Resizing and Normalization**
# 
# Before we feed our images to network, we should be resizing it to the desired size. Moreover, to improve model performance, we should normalize the image pixel value (keeping them in range 0 and 1 by dividing by 256). This should happen while training as well as inference. Hence we can add that as a layer in our Sequential Model.
# 
# You might be thinking why do we need to resize (256,256) image to again (256,256). You are right we don't need to but this will be useful when we are done with the training and start using the model for predictions. At that time somone can supply an image that is not (256,256) and this layer will resize it

# In[20]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# **Data Augmentation**
# 
# Data Augmentation is needed when we have less data, this boosts the accuracy of our model by augmenting the data.
# 
# 

# In[21]:


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.4), 
])


# In[22]:


train_ds_augmented = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# **Model Architecture**
# 
# We use a CNN coupled with a Softmax activation in the output layer. We also add the initial layers for resizing, normalization and Data Augmentation.
# 
# **We are going to use convolutional neural network (CNN) here. CNN is popular for image classification tasks.**
# 
# 

# In[23]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

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


# In[24]:


model.summary()


# **Compiling the Model**
# 
# We use **adam** Optimizer, **SparseCategoricalCrossentropy** for losses, **accuracy** as a metric

# In[25]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,  # Adjust the patience parameter
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


history = model.fit(
    train_ds_augmented,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# In[28]:


scores = model.evaluate(test_ds)


# In[29]:


score = model.evaluate(train_ds)


# **You can see above that we get 99.61% accuracy for our test dataset. This is considered to be a good accuracy**

# In[30]:


scores


# Scores is just a list containing loss and accuracy value

# **Plotting the Accuracy and Loss Curves**

# In[31]:


tf.keras.callbacks.History()


# In[32]:


history.params


# In[33]:


history.history.keys()


# **loss, accuracy, val loss etc are a python list containing values of loss, accuracy etc at the end of each epoch**

# In[34]:


type(history.history['loss'])


# In[35]:


len(history.history['loss'])


# In[36]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[40]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[41]:


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


# **Run prediction on a sample image**

# In[43]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    


# **Write a function for inference**

# In[44]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# **Now run inference on few sample images**

# In[45]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i].numpy()]

        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[ ]:


pip install scikit-learn


# In[54]:


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Assuming you have a list to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Collect all unique class names from the training dataset
all_class_names = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']
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


# In[55]:


# Calculate precision with zero_division parameter
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print(f'Precision: {precision:.2f}')


# In[56]:


# Calculate recall with zero_division parameter
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print(f'Recall: {recall:.2f}')


# In[57]:


# Calculate precision with zero_division parameter
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
print(f'F1-score: {f1:.2f}')


# In[73]:


pip install seaborn


# In[58]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have true_labels and predicted_labels

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Define class names for visualization
class_names = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Qualit_Fruits']

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# **Saving the Model**

# In[59]:


model.save("D:\Project\MODELS2\Fruit_quality.h5")


# In[15]:


keras_model = tf.keras.models.load_model("D:\Project\MODELS2\Fruit_quality.h5")


# In[ ]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# In[65]:


with open("D:\Project\MODELS\Fruit_quality.tflite", "wb") as tflite_file:
    tflite_file.write(tflite_model)


# In[ ]:




