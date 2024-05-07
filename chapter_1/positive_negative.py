import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create 2-depth neural network model

# Sequential type network which layer sequentially
model = Sequential()

# Add first layer. accept 1 input, 8 output(number of neuron in layer)
model.add(Dense(units=8, activation='relu', input_dim=1))

# Add second layer. accept 1 output because this is last layer
model.add(Dense(units=1, activation='sigmoid'))

# Set loss function type. optimizer type
model.compile(loss='mean_squared_error', optimizer='sgd')

# TensorFlow's model.fit() method expects the input data x to be in the form of a NumPy array or a TensorFlow tensor

# Create train set x, y. y is correspondence of x. As you can see, negative values correspond to 0
x_train = np.array([1, 2, 3, 10, 20, -2, -10, -100, -5, -20])
y_train = np.array([1.0, 1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0])

# Begin training. epochs is number of train, batch_size is frequency of weighting updates
model.fit(x_train, y_train, epochs=10, batch_size=4)

# Create test set x, y. use model.predict to infer test_x
test_x = np.array([30, 40, -20, -60])
test_y = model.predict(test_x)

# Print resulf of inference
for i in range(len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], test_y[i]))



"""
Result

Epoch 1/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.5126 
Epoch 2/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 579us/step - loss: 0.6516
Epoch 3/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 562us/step - loss: 0.4612
Epoch 4/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 561us/step - loss: 0.5449
Epoch 5/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 629us/step - loss: 0.5001
Epoch 6/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 595us/step - loss: 0.4894
Epoch 7/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 652us/step - loss: 0.4817
Epoch 8/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 540us/step - loss: 0.4688
Epoch 9/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 641us/step - loss: 0.5333
Epoch 10/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 551us/step - loss: 0.5890
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
input 30 => predict: [0.9909521]
input 40 => predict: [0.9980837]
input -20 => predict: [0.9999883]
input -60 => predict: [1.]
"""