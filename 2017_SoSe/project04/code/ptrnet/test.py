from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq, Pointer
from keras.utils.test_utils import keras_test
from pprint import pprint
import keras.backend as K
import numpy as np
import utils

input_length = 5
input_dim = 3

output_length = input_length
output_dim = input_length

samples = 10
hidden_dim = 7

BS = 1


'''
TSP Problem parameters
'''
tsp_input_dim = 2
tsp_samples = 1000
tsp_hidden_dim = 10
tsp_input_length = 10
tsp_output_dim = tsp_input_length           #Due to one-hot encoding.
tsp_output_length = tsp_input_length


@keras_test
def test_SimpleSeq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))

    models = []
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim))]
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]

    for model in models:
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, nb_epoch=1)


@keras_test
def test_Seq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))

    models = []
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim))]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True)]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2)]

    # for model in models:
    #     model.compile(loss='mse', optimizer='sgd')
    #     model.summary()
    #     model.fit(x, y, epochs=1)

    model = Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2, teacher_force=True)
    model.compile(loss='mse', optimizer='sgd')
    model.fit([x, y], y, epochs=1)
    for layer in model.layers:
        print layer
        if "RecurrentSequential" in str(layer):
            print K.get_value(layer.x)
            # print K.



    
@keras_test
def test_AttentionSeq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))

    models = []
    models += [AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim))]
    models += [AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
    models += [AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=3)]

    # att = AttentionSeq2Seq(output_dim=tsp_output_dim, hidden_dim=tsp_hidden_dim, output_length=tsp_output_length, input_shape=(tsp_input_length, tsp_input_dim))

    for model in models:
        model.compile(loss='mse', optimizer='sgd')
        print model.summary()
        model.fit(x, y, epochs=1)

@keras_test
def test_PointerSeq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = []
    for i in range(samples):
        ar = []
        for j in range(input_length):

            arr = np.zeros(input_length)
            index = np.random.randint(input_dim)
            arr[index] = 1
            ar.append(arr)
        y.append(ar)
    y = np.asarray(y)

    print "Done making dummy data"
    models = Pointer(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim))
    # models += [Pointer(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
    # models += [Pointer(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=3)]
    print "Done creating model"

    # models.compile(loss='mse', optimizer='fast_compile')
    models.compile(loss='mse', optimizer='sgd')
    print models.summary()
    models.fit(x, y, epochs=1)

    print "Done everything master"
    while True:
        cmd = raw_input("Master, please give Dobby a sock now. (Just write sock)")
        if cmd.lower() == "sock":
            break
        print "Master, why must thy be so cruel."   
        print "Let's try that again."

@keras_test
def test_PointerSeq2Seq_TSP():
    ''' 
        The data has been generated from https://github.com/vyraun/Keras-Pointer-Network.
            A sample in X looks like [(8, 0), (2, 8), (6, 9), (9, 8), (7, 5), (0, 5), (4, 6), (8, 2), (5, 2), (4, 9), (5, 0)]
            A sample in Y looks like [0, 1, 3, 9, 7, 8, 5, 4, 2, 10, 6]
    '''
    X = []
    Y = []
    for _ in xrange(0,tsp_samples):
        X.append(utils.generate_data(tsp_input_length))
    for samples in X:
        solution = utils.Tsp(samples)
        Y.append(solution.solve_tsp_dynamic())

    '''
        One hot encoding for the output symbols.
    '''
    one_hot_matrix = np.eye(tsp_input_length)
    Y = [[ one_hot_matrix[sample[x]] for x in range(len(sample)) ] for sample in Y ]

    # pprint(X[0])
    # pprint(Y[0])
    # raw_input()

    #Transmuting the data into Numpy arrays
    X = np.asarray(X)/10.0
    Y = np.asarray(Y)

    x_train,x_test = X[:int(X.shape[0]*.80)],X[int(X.shape[0]*.80):]
    y_train,y_test = Y[:int(Y.shape[0]*.80)],Y[int(Y.shape[0]*.80):]
        
    print "Done making dummy data"
    print "tsp_input_length", tsp_input_length, "sd", tsp_input_dim
    models = Pointer(output_dim=tsp_output_dim, hidden_dim=tsp_hidden_dim, output_length=tsp_output_length, input_shape=(tsp_input_length, tsp_input_dim), batch_size=10,bidirectional=False)
    print "Done creating model"

    # models.compile(loss='mse', optimizer='fast_compile')
    models.compile(loss='mse', optimizer='sgd')
    print models.summary()
    models.fit(X, Y, epochs=10,batch_size=10)
    print "Done fitting model"

    print "Done everything master"
    while True:
        cmd = raw_input("Master, please give Dobby a sock now. (Just write sock)")
        if cmd.lower() == "sock":
            break
        print "Master, why must thy be so cruel."   
        print "Let's try that again."

if __name__ == "__main__":
    test_PointerSeq2Seq_TSP()