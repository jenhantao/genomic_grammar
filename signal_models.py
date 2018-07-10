from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.layers import *
from keras.models import Model
from keras import backend as K
import keras
import tensorflow as tf

def root_mean_squared_log_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)), axis=-1)) 

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def pearson_correlation(y_true, y_pred):
    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)
    diff_true = y_true - mean_true
    diff_pred = y_pred - mean_pred
    
    
    diff_true_squared = tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_true)))
    diff_pred_squared = tf.sqrt(tf.reduce_sum(tf.square(y_pred - mean_pred)))
    correlation = tf.reduce_sum(diff_true * diff_pred)/(diff_true_squared * diff_pred_squared)
    return correlation
    
def get_additiveAttention_model(total_seq_length,
    mode,
    num_classes = 2,
    num_motifs=32, 
    motif_size=10,
    adjacent_bp_pool_size=10,
    attention_dim=10,
    attention_hops=1,
    dropout_rate=0.1):

    # set model training settings
    if mode == 'classification':
        mode_activation = 'sigmoid'
        mode_loss = keras.losses.categorical_crossentropy
        mode_metrics = ['categorical_accuracy']
        mode_optimizer = keras.optimizers.Adam()

    elif mode == 'signal_regression':
        mode_activation = 'relu'
        mode_loss = keras.losses.mean_squared_logarithmic_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()

    elif mode == 'fold_regression':
        mode_activation = 'linear'
        mode_loss = keras.losses.mean_absolute_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()
    
    input_fwd = Input(shape=(total_seq_length,4), name='input_fwd')

    ### find motifs ###
    convolution_layer = Conv1D(filters=num_motifs, 
        kernel_size=motif_size,
        activation='relu',
        input_shape=(total_seq_length,4),
        name='convolution_layer',
        padding = 'same',
        use_bias = False
        )
    forward_motif_scores = convolution_layer(input_fwd)
    ### attention tanh layer ###
    attention_tanh_layer = Dense(attention_dim,
        activation='tanh',
        use_bias=False,
        name = 'attention_tanh_layer')
    attention_tanh_layer_out = attention_tanh_layer(forward_motif_scores)

    ### outer layer ###
    attention_outer_layer = Dense(attention_hops,
        activation='linear',
        use_bias=False,
        name = 'attention_outer_layer')
    attention_outer_layer_out = attention_outer_layer(attention_tanh_layer_out)

    
    ### apply softmax ###
    softmax_layer = Softmax(axis=1, name='attention_softmax_layer')
    attention_softmax_layer_out = softmax_layer(attention_outer_layer_out)

    ### attention dropout ###
    attention_dropout_layer = Dropout(dropout_rate, name='attention_dropout')
    attention_dropout_layer_out = attention_dropout_layer(attention_softmax_layer_out)
    
    ### attend to hidden states ###
    attending_layer = Dot(axes=(1,1),
        name='attending_layer')

    attended_states = attending_layer([attention_dropout_layer_out, forward_motif_scores])

#    dense_layer = TimeDistributed(Dense(units=1, activation = 'linear'),
#                                  name='dense_layer')
#    dense_out = dense_layer(attended_states)
    
    # make prediction
    flattened = Flatten(name='flatten')(attended_states)
    predictions = Dense(num_classes,
                        name='predictions',
                        activation = mode_activation
                       )(flattened)

    # define and compile model
    model = Model(inputs=[input_fwd], outputs=predictions)

    model.compile(loss=mode_loss,
                  optimizer=mode_optimizer,
                  metrics=mode_metrics)
    return model

def element_multiply (x,y):
    x_shape = []
    for i, s in zip(K.int_shape(x), tf.unstack(tf.shape(x))):
        if i is not None:
            x_shape.append(i)
        else:
            x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(K.int_shape(y), tf.unstack(tf.shape(y))):
        if i is not None:
            y_shape.append(i)
        else:
            y_shape.append(s)
    y_shape = tuple(y_shape)

    xt = tf.reshape(x, [-1, x_shape[-1],1])
    yt = tf.reshape(y, [y_shape[-2],1])

    return tf.multiply(xt,yt)

class Projection(Layer):
    """
    Learn linear transform of imput tensor
    """
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activations.linear
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        super(Projection, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super(Projection, self).build(input_shape)

    def call(self, inputs):
        output = element_multiply(inputs, self.kernel)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        return config

    def compute_output_shape(self, input_shape):
        output_shape = (self.units, input_shape[1])
        return output_shape

def get_dotProductAttention_model(total_seq_length,
    mode,
    num_classes = 1,
    num_motifs=150, 
    motif_size=10,
    adjacent_bp_pool_size=10,
    num_dense_neurons=10,
    dropout_rate=0.75):
    
    # set model training settings
    if mode == 'classification':
        mode_activation = 'sigmoid'
        mode_loss = keras.losses.categorical_crossentropy
        mode_metrics = ['categorical_accuracy']
        mode_optimizer = keras.optimizers.Adam()

    elif mode == 'signal_regression':
        mode_activation = 'relu'
        mode_loss = keras.losses.mean_squared_logarithmic_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()

    elif mode == 'fold_regression':
        mode_activation = 'linear'
        mode_loss = keras.losses.mean_absolute_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()
    
    input_fwd = Input(shape=(total_seq_length,4), name='input_fwd')

    ### find motifs ###
    convolution_layer = Conv1D(filters=num_motifs, 
        kernel_size=motif_size,
        activation='relu',
        input_shape=(total_seq_length,4),
        name='convolution_layer',
        padding = 'same',
        use_bias=False,
        )
    forward_motif_scores = convolution_layer(input_fwd)
    ### crop motif scores to avoid parts of sequence where motif score is computed in only one direction ###

    forward_motif_scores = forward_motif_scores
    ### pool across length of sequence ###
    sequence_pooling_layer = MaxPool1D(pool_size=adjacent_bp_pool_size, 
        strides=adjacent_bp_pool_size,
        name='sequence_pooling_layer')
    pooled_scores = sequence_pooling_layer(forward_motif_scores)
        
    ### compute attention ###
    ### weight queries ###
    query_transformer = TimeDistributed(Projection(units=1), 
                                        input_shape=(int(total_seq_length/adjacent_bp_pool_size), num_motifs*2),
                                        name='query_transformer'
                                       )
    weighted_queries = query_transformer(pooled_scores)
    
    ### weight keys ###
    key_transformer = TimeDistributed(Projection(units=1), 
                                      input_shape=(int(total_seq_length/adjacent_bp_pool_size), num_motifs*2),
                                      name = 'key_transformer')
    weighted_keys = key_transformer(pooled_scores)
    
    dot_product = Dot(axes=(2,2),name='dot_product')
    attention_weights = dot_product([weighted_queries, weighted_keys])

    #scaling_layer = Lambda(lambda x: x/(int(num_motifs*2)**-2),
    #    name='scaling_layer')
    #scaled_attention_weights = scaling_layer(attention_weights)

    ### apply softmax ###
    softmax_layer = Softmax(axis=1, name='attention_softmax_layer')
    attention_softmax_layer_out = softmax_layer(attention_weights)
    #attention_softmax_layer_out = softmax_layer(scaled_attention_weights)
    
    attention_dropout_layer = Dropout(dropout_rate, name='attention_dropout')
    attention_dropout_layer_out = attention_dropout_layer(attention_softmax_layer_out)
    
    ### weight values ###
    value_transformer = TimeDistributed(Projection(units=1), 
                                        input_shape=(int(total_seq_length/adjacent_bp_pool_size), num_motifs*2),
                                        name='value_transformer'
                                       )
    
    weighted_values = value_transformer(pooled_scores)
    
    ### attend to hidden states ###
    ax1 = 1
    ax2 = 1
    attending_layer = Dot(axes=(ax1,ax2),
        name='attending_layer')
    #print('attending axes', ax1,ax2, 'linear')
    attended_states = attending_layer([attention_dropout_layer_out, weighted_values])

    # make prediction
    dense_layer = TimeDistributed(
        Dense(
        units=num_dense_neurons, 
        activation = 'tanh'),
        name='dense_layer')
    dense_out = dense_layer(attended_states)

    flattened = Flatten(name='flatten')(dense_out)#(drop_out)

    predictions = Dense(num_classes,
                        name='predictions',
                        activation = mode_activation, 
                       )(flattened)

    # define and compile model
    model = Model(inputs=[input_fwd], outputs=predictions)

    model.compile(loss=mode_loss,
                  optimizer=mode_optimizer,
                  metrics=mode_metrics)
    return model
    

def get_convolution_model(
    total_seq_length,
    mode,
    num_classes = 1,
    num_motifs = 150,
    motif_size = 10,
    num_dense_neurons = 50, 
    dropout_rate = 0.75
    ):
    '''
    Implementation of DeepBind model adapted to also do regression
    in addition to classification of regulatory sequences (enhancers)
    '''
    
    # set model training settings
    if mode == 'classification':
        mode_activation = 'sigmoid'
        mode_loss = keras.losses.categorical_crossentropy
        mode_metrics = ['categorical_accuracy']
        mode_optimizer = keras.optimizers.Adam()

    elif mode == 'signal_regression':
        mode_activation = 'relu'
        mode_loss = keras.losses.mean_squared_logarithmic_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()

    elif mode == 'fold_regression':
        mode_activation = 'linear'
        mode_loss = keras.losses.mean_absolute_error
        mode_metrics = [pearson_correlation]
        mode_optimizer = keras.optimizers.RMSprop()
    else:
        return None

    input_fwd = Input(shape=(total_seq_length,4), name='input_fwd')
    input_rev = Input(shape=(total_seq_length,4), name='input_rev')

    # find motifs
    convolution_layer = Conv1D(filters=num_motifs, 
        kernel_size=motif_size,
        activation='relu',
        input_shape=(total_seq_length,4),
        name='convolution_layer',
        padding = 'same',
        use_bias = False,
        )
    forward_motif_scores = convolution_layer(input_fwd)
    reverse_motif_scores = convolution_layer(input_rev)


    # calculate max scores for each orientation
    seq_pool_layer = MaxPool1D(pool_size=total_seq_length)
    max_fwd_scores = seq_pool_layer(forward_motif_scores)
    max_rev_scores = seq_pool_layer(reverse_motif_scores)

    # calculate max score for strand
    orientation_max_layer = Maximum()
    max_seq_scores = orientation_max_layer([max_fwd_scores, max_rev_scores])

    # fully connected layer
    dense_out = Dense(num_dense_neurons, activation='relu', 
                     )(max_seq_scores)

    # drop out
    drop_out = Dropout(dropout_rate)(dense_out)

    # make prediction
    flattened = Flatten()(drop_out)
    predictions = Dense(num_classes,
                        activation = mode_activation, 
                       )(flattened)
    
    # define and compile model
    model = Model(inputs=[input_fwd, input_rev], outputs=predictions)

    model.compile(loss=mode_loss,
                  optimizer=mode_optimizer,
                  metrics=mode_metrics)
    return model
