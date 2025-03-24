import tensorflow as tf
import sklearn
print(tf.__version__)
print(sklearn.__version__)

import tensorflow as tf
import tensorflow_datasets as tfds

assert tf.__version__.startswith('2.')

# Load the Iris dataset 
(ds_train, ds_test), ds_info = tfds.load(
    'iris',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,  # Load the dataset as (feature, label) pairs
    with_info=True       # Retrieve the metadata of the dataset
)

# Function to preprocess the dataset by one-hot encoding the labels
def preprocess(features, label):
    label = tf.one_hot(label, depth=3)  # 3 classes in the Iris dataset
    return features, label

# Prepare the train and test datasets
train_data = ds_train.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_data = ds_test.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Define a simple neural network for classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),  # Input shape is 4 for the four Iris features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Softmax for multi-class classification
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data and validate on the test data
history = model.fit(train_data, epochs=50, validation_data=test_data)

# Evaluate the model's performance on the test dataset
loss, accuracy = model.evaluate(test_data)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Use the model to make predictions on a new sample
sample_data = tf.constant([[5.1, 3.3, 1.7, 0.5]])  # Example measurements of an iris flower

# Make the predictions
predictions = model.predict(sample_data)
predicted_class = tf.argmax(predictions, axis=1).numpy()

print(f"Predicted class: {predicted_class}")

# Explore the dataset
for features, label in ds_train.take(1):  # Adjust this part based on whether ds_train is batched
    print("Features:", features.numpy())  # This will print the features of one batch or one example
    print("Label:", label.numpy())        # This will print the labels of one batch or one example
    
    import tensorflow as tf

def normalize_data(features, label):
    """Normalize the features based on dataset statistics.
    Assumes features are in the format [sepal length, sepal width, petal length, petal width].
    """
    # Define mean and standard deviation for the Iris dataset features
    mean = tf.constant([5.84, 3.05, 3.76, 1.20], dtype=tf.float32)
    std = tf.constant([0.83, 0.43, 1.76, 0.76], dtype=tf.float32)
    
    # Normalize features
    features = (features - mean)/std
    return features, label

# Specify batch size for training and testing
batch_size = 32

# Assuming ds_train and ds_test are previously defined TensorFlow Dataset objects
# Apply normalization, shuffle training data, and batch both datasets
ds_train = ds_train.map(normalize_data).shuffle(1000).batch(batch_size)
ds_test = ds_test.map(normalize_data).batch(batch_size)
