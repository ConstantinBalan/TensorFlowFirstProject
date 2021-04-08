import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
#Split data into testing and training data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Import class names attributed to the objects in the test data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
train_images = train_images / 255.0
test_images = test_images / 255.0
#A one layer NN using a Sequential model because of only one input and output tensor
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #Shape the input to be 28x28px to represent the image being passed in
    tf.keras.layers.Dense(128, activation='relu'), #Uses a dense layer with 128 neurons with the relu(rectified linear) activation function
    tf.keras.layers.Dense(10)#Output layer with 10 neurons corresponding to each class name pointed out above
])
#Training model using adam optimizer, SparseCategoricalCrossEntropy due to multiple labels, and outputting the accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Pass in the training data, and set a number of epochs for the model to test in
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')