import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.pyplot.switch_backend('Agg') 

# CONSTANTS
CLIENT_NUM = 2 # Should be less than 10
ROUND = 0

# Load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape dataset
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  
x_test /= 255

# Truncate dataset
dataset_length = 60000
y_train = y_train[:dataset_length]
x_train = x_train[:dataset_length]

# Split Dataset
x_train_1 = []
y_train_1 = []
x_train_2 = []
y_train_2 = []
for i in range(len(x_train)):
    if y_train[i] <= 4:
        x_train_1.append(x_train[i])
        y_train_1.append(y_train[i])
    else:
        x_train_2.append(x_train[i])
        y_train_2.append(y_train[i])
x_train_1 = np.array(x_train_1)
y_train_1 = np.array(y_train_1)
x_train_2 = np.array(x_train_2)
y_train_2 = np.array(y_train_2)

client_x_trains = [x_train_1, x_train_2]
client_y_trains = [y_train_1, y_train_2]


# FedAVG
def getAveragedWeight(model_weight, n, n_k_set):
  new_weights = [np.zeros(k.shape) for k in model_weight[0]]
  for c in range(len(model_weight)):
    for i in range(len(new_weights)):
        new_weights[i] += (n_k_set[c] * model_weight[c][i] / n)

  return new_weights

# Create Model
TNN_G = keras.Sequential([
    keras.layers.Dense(200, activation='relu', input_shape=(784,)),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(10)
])
TNN_G.compile(optimizer='sgd',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

def plotResult(rounds, acc):
    plt.ylim(top=1.00)
    plt.plot(rounds, acc, marker = 'o')
    plt.title("MNIST 2NN Non-IID (K = 2)")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.savefig('history.png')


from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

app = Flask(__name__);
CORS(app, resources={r"/*": {"origins": "*"}})

CONNECTED_CLIENT_NUM = -1
@app.route('/createClient', methods=['GET'])
def create_client():
    global CONNECTED_CLIENT_NUM

    CONNECTED_CLIENT_NUM += 1
    print(f'New client connected. Total num: {CONNECTED_CLIENT_NUM+1}')
    if CONNECTED_CLIENT_NUM < CLIENT_NUM:
        return jsonify(
                {'client_id': CONNECTED_CLIENT_NUM, 'x_train': client_x_trains[CONNECTED_CLIENT_NUM].tolist(), 'y_train': client_y_trains[CONNECTED_CLIENT_NUM].tolist()},
            )
    else:
        return jsonify(
                {'client_id': 0, 'x_train': client_x_trains[0].tolist(), 'y_train': client_y_trains[0].tolist()},
            )

COMROUND_HISTROY = []
ACCURACY_HISTORY = []
RECEIVED_WEIGHTS = []
RECEIVED_LENGTHS = []
GLOBAL_WEIGHT = None
@app.route('/postWeights', methods=['POST'])
def post_weights():
    global ROUND

    data = request.json
    w = data['model_weight']
    l = data['train_length']
    # Conver list to ndarray
    def convert_weights_to_tensor(arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg.numpy()
    w = [convert_weights_to_tensor(i) for i in w]

    RECEIVED_WEIGHTS.append(w)
    RECEIVED_LENGTHS.append(l)

    print(f'New weight received. Total num: {len(RECEIVED_WEIGHTS)}')
    def performFedAvg():
        global ROUND, GLOBAL_WEIGHT, RECEIVED_WEIGHTS, RECEIVED_LENGTHS, ACCURACY_HISTORY, COMROUND_HISTROY
        print(f'Averaging Round: {ROUND}')
        GLOBAL_WEIGHT = getAveragedWeight(model_weight=RECEIVED_WEIGHTS, n=dataset_length, n_k_set=RECEIVED_LENGTHS)
        RECEIVED_WEIGHTS = []
        RECEIVED_LENGTHS = []
        TNN_G.set_weights(GLOBAL_WEIGHT)
        result = TNN_G.evaluate(x_test, y_test)
        COMROUND_HISTROY.append(ROUND)
        ACCURACY_HISTORY.append(result[1])
        plotResult(COMROUND_HISTROY, ACCURACY_HISTORY)

    if len(RECEIVED_WEIGHTS) == CLIENT_NUM:
        ROUND += 1
        thread = threading.Thread(target=performFedAvg)
        thread.start()

    return jsonify(
            {'is_received': True},
        )


@app.route('/getGlobalWeights', methods=['GET'])
def get_weights():
    if GLOBAL_WEIGHT != None:
        weight = [i.tolist() for i in GLOBAL_WEIGHT]
        print(f'Global weight sent.')
        return jsonify({'isAveraged': True, 'global_weight': weight})
    else:
        return jsonify({'isAveraged': False, 'global_weight': []})

app.run(debug=True, host='172.20.10.2')