import numpy as np

# transform and load data from a csv file
# keras LSTM layer works by taking a numpy array of 3 dimensions (N, W, F)
# wehere N is the number of training sequences, W the sequence length and F the number of featrure of each sequence
def load_data(filename, seq_len, normalise_window):
    # open file
    f = open(filename, 'rb').read()
    # decode the file and split it where a new line is found
    data = f.decode().split('\n')
    sequence_length = seq_len + 1
    result = []
    # for each index
    for index in range(len(data) - sequence_length):
        # 'reshape' the data. each one considers from index + the sequence lenght (i.e. the window size)
        result.append(data[index: index + sequence_length])
    # convert the resulting array 
    result = np.array(result)
    # split test/train data. consider 90% of data for train
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]