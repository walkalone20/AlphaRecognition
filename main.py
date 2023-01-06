from multiprocessing import pool
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv')

feature = data.iloc[:, data.columns != '0'].values
label = data[data.columns[0]].values

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train / 255.
x_test = x_test / 255.

print(x_train.shape)

model = Sequential([
    Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation="relu"), 
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), 
    Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(26, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

def show_example(index, preds, dsx, dsy):
    current_img = dsx[index][:, :, 0] * 255
    prediction = np.argmax(preds[index])
    if(len(dsy) > 0 or dsy != None) :
        label = dsy[index]
        print("Label:", label)
    print("Prediction:", prediction)
    plt.imshow(current_img, interpolation='nearest', cmap='gray')
    plt.show()

# show_example(9, preds, x_test, y_test), show_example(23, preds, x_test, y_test)

def analysis(preds, limit, dsx, dsy):
    correct = 0
    misclassified = []
    for i in range(limit):
        prediction = np.argmax(preds[i])
        label = dsy[i]
        if(prediction == label):
            correct += 1
        else:
            misclassified.append(i)
    
    print(f"Predictions in a limit of {limit} are {(correct / limit) * 100} correct")
    print(f"Misclassfied {len(misclassified)} examples:")
    for i in misclassified:
        show_example(i, preds, dsx, dsy)

preds = model.predict(x_test)
analysis(preds, x_test.shape[0], x_test, y_test)

def predict(filepath, submitfile):
    X_test = pd.read_csv(filepath)  # 测试数据存放于 test.cv

    X_test = X_test.to_numpy()
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test / 255.
    preds = model.predict(X_test)
    labels = [np.argmax(i) for i in preds]
    idxs = [i+1 for i in range(len(labels))]
    submit = pd.DataFrame({'ImageId': idxs, 'Label': labels})  # 格式
    submit.to_csv(submitfile, index=False)
