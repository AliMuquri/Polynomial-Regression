import tensorflow as tf
import os
from PolynomialRegression import PolynomialRegression
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau as rlp
from tensorflow.keras.callbacks import TensorBoard



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#tf.config.experimental_run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

main_path = os.getcwd()
dataset_path = os.path.join(main_path, 'dataset')
train_df_path = os.path.join(dataset_path, 'traintestDf.csv')


def infer_data_types(cols):
    ''' dynmically infering datatype by providing a synthetic sample
    Args:
    cols: a row containing every column value to be typed
    Returns:
    datatypes: a list of datatypes/samples for every column'''

    datatypes = []
    for col in cols:

        try:
            tf.strings.to_number(col)
            datatypes.append(tf.constant(0.0, dtype=tf.float32))
        except tf.errors.InvalidArgumentError:
            try:
                tf.strings.to_number(col, out_type=tf.int32)
                datatypes.append(tf.constant(0, dtype=tf.float32))
            except tf.errors.InvalidArgumentError:
                datatypes.append(tf.constant(0, dtype=tf.float32))
    return datatypes

def split_feature_label(*data):
    ''' split the data into features and labels
    Args:
    data: dataset
    Returns:
    data_tuple: a tuple containing the features and label
    '''
    features, label = list(data[:-1]), data[-1]
    #need to turn a list of tensor features into one big tensor
    features = tf.stack(features, axis=0)
    size  = tf.shape(features)[0]
    features = tf.reshape(features, [1, size])
    
    data_tuple = (features, label)
    #print(data_tuple)
    return data_tuple

def dynamically_split_val_train(dataset_gen,  split_portion=0.2):
    ''' split the dataset into train and validation
    Args:
    dataset_gen: the entire dataset
    split_portion: the split proportion used for splitting
    Returns:
    train_gen: the dataset for traning
    validation_gen: the dataset for validation '''
    dataset_size = dataset_gen.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
    number_datapoints = int(dataset_size * split_portion)
    dataset_gen = dataset_gen.shuffle(buffer_size=number_datapoints, reshuffle_each_iteration=True)
    train_gen = dataset_gen.skip(number_datapoints)
    validation_gen =  dataset_gen.take(number_datapoints)
    return train_gen, validation_gen

def main():
    dataset_gen = tf.data.TextLineDataset(train_df_path)
    dataset_gen = dataset_gen.skip(1)
    
    split_lines = tf.strings.split(next(iter(dataset_gen)), ',')
    input_dim = int(tf.shape(split_lines)[0].numpy().tolist()-1)
    column_defaults = infer_data_types(split_lines)

    dataset_gen = dataset_gen.map(lambda line: tf.io.decode_csv(line, record_defaults=column_defaults), num_parallel_calls=tf.data.AUTOTUNE)
    dataset_gen = dataset_gen.map(split_feature_label, num_parallel_calls=tf.data.AUTOTUNE)    
    
    train_gen, val_gen = dynamically_split_val_train(dataset_gen)
   
    train_gen = train_gen.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_gen = val_gen.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    rlp_obj = rlp('loss', factor=0.8, patience=2,  min_lr =1e-10)    
    tb = TensorBoard(log_dir='log', histogram_freq=1)
    #tf.profiler.experimental.start(logdir='log' , options=tf.profiler.experimental.ProfilerOptions(host_tracer_level=1, python_tracer_level = 1 ))
    model = PolynomialRegression(input_dim, degree=3.0)
    model.build((None,input_dim))
    
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-2, clipnorm=1.0), loss=mse_loss(), metrics=[mse_metric()])
    model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[rlp_obj,tb])
    
    #tf.profiler.experimental.stop()

if __name__ == "__main__":
    main()
