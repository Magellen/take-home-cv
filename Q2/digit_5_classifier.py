import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def run():
    num_classes = 2
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = (y_train==5).astype(np.uint8)
    y_test = (y_test==5).astype(np.uint8)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    batch_size = 128
    epochs = 1
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def plot_p_r(precisions,recalls):
        plt.figure(figsize=(12,8))
        plt.title('Precisions versus recall')
        plt.plot(precisions[:-1],recalls[:-1])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.savefig("PR.jpg")
    
    y_scores = model.predict(x_test)
    y_scores = y_scores[:,1]
    y_test = y_test.argmax(1)
    precisions,recalls,thresholds = precision_recall_curve(y_test, y_scores)
    plot_p_r(precisions,recalls)



if __name__ == "__main__":   
    run()

