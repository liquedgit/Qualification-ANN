import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()

def loadData():
    df = pd.read_csv('./NFLX.csv')
    feature = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    target = df[['Close']]
    return feature, target

feature, target = loadData()

converter = OrdinalEncoder()
feature_data = converter.fit_transform(feature)

scaler = MinMaxScaler()
feature_data = scaler.fit_transform(feature_data)

target_data = scaler.fit_transform(target)

layers = {
    'input': 5,
    'hidden1': 16,
    'hidden2': 8,
    'output': 1
}

inp_hidden1 = {
    'weight': tf.Variable(tf.random.normal([layers['input'], layers['hidden1']])),
    'bias': tf.Variable(tf.random.normal([layers['hidden1']]))
}

hidden1_hidden2 = {
    'weight': tf.Variable(tf.random.normal([layers['hidden1'], layers['hidden2']])),
    'bias': tf.Variable(tf.random.normal([layers['hidden2']]))
}

hidd2_out = {
    'weight': tf.Variable(tf.random.normal([layers['hidden2'], layers['output']])),
    'bias': tf.Variable(tf.random.normal([layers['output']]))
}

feature = tf.placeholder(tf.float32, [None, layers['input']])
target = tf.placeholder(tf.float32, [None, layers['output']])

def feed_forward(inp_dataset):
    x1 = tf.matmul(inp_dataset, inp_hidden1['weight']) + inp_hidden1['bias']
    y1 = tf.nn.relu(x1)
    
    x2 = tf.matmul(y1, hidden1_hidden2['weight']) + hidden1_hidden2['bias']
    y2 = tf.nn.relu(x2)
    
    x3 = tf.matmul(y2, hidd2_out['weight']) + hidd2_out['bias']
    y3 = tf.nn.sigmoid(x3)
    
    return y3

output = feed_forward(feature)
error = tf.reduce_mean(0.5 * (target - output) ** 2)

learning_rate = 0.1
train = tf.train.AdamOptimizer(learning_rate).minimize(error)

x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.2)

epoch = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch + 1):
        train_dict = {
            feature: x_train,
            target: y_train
        }
        
        sess.run(train, feed_dict=train_dict)
        
        loss = sess.run(error, feed_dict=train_dict)
        
        if i % 1000 == 0:
            print('Iteration: {}, current error: {}'.format(i, loss))
            
    accuracy = tf.reduce_mean(tf.abs(target - output))
    
    test_dict = {
        feature: x_test,
        target: y_test
    }
    print("Accuracy: {}%".format(100 - sess.run(accuracy, feed_dict=test_dict) * 100))

    predictions = sess.run(output, feed_dict=test_dict)
    
    predicted_values = scaler.inverse_transform(predictions)

    print("Predicted close prices: ")
    print(predicted_values[0][0])
