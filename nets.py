from definitions import *
from definitionskeras import *


def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# Function to create a convolutional layer
def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters,
                               _padding='SAME',
                               maxpool=True
                               ):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=_padding)
    layer += biases
    ## We shall be using max-pooling.
    if maxpool:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding=_padding)

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer


# Function to create a Flatten Layer
def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()
    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()
    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer


# Function to create a Fully - Connected Layer
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True,
                    dropoutRate=0.2
                    ):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    dropped = tf.nn.dropout(input, rate=dropoutRate)
    layer = tf.matmul(dropped, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def CNNmodelKeras(img_size, num_channels, num_classes, type):
    # Architecture =================================================================================
    filter_size_conv1 = 3
    num_filters_conv1 = 32
    filter_size_conv2 = 3
    num_filters_conv2 = 64
    filter_size_conv3 = 3
    num_filters_conv3 = 128
    filter_size_conv4 = 3
    num_filters_conv4 = 256
    fc_layer_size = 256

    dropout_rate = 0.2

    inputs = Input(shape=(img_size, img_size, num_channels))
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    x = Dropout(dropout_rate)(x)  # Added dropout after the first max pooling layer

    x = Conv2D(num_filters_conv2, kernel_size=(filter_size_conv2, filter_size_conv2), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    x = Dropout(dropout_rate)(x)  # Added dropout after the second max pooling layer

    x = Conv2D(num_filters_conv3, kernel_size=(filter_size_conv3, filter_size_conv3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    x = Dropout(dropout_rate)(x)  # Added dropout after the third max pooling layer

    x = Flatten()(x)

    if type == 'discrete':
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Added dropout after the first fully connected layer
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Added dropout after the second fully connected layer
        x = Dense(num_classes)(x)
        x = Softmax()(x)

    if type == 'continuous':
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Added dropout after the first fully connected layer
        x = Dense(1)(x)

    saliency_model = Model(inputs=inputs, outputs=x)
    return saliency_model

def CNNDeepmodelKeras(img_size, num_channels, num_classes, type):
    # Architecture =================================================================================
    filter_size_conv1 = 5
    num_filters_conv1 = 64
    filter_size_conv2 = 3
    num_filters_conv2 = 64
    filter_size_conv3 = 3
    num_filters_conv3 = 128
    filter_size_conv4 = 3
    num_filters_conv4 = 256
    fc_layer_size = 128
    inputs = Input(shape=(img_size, img_size, num_channels))

    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters_conv1, kernel_size=(filter_size_conv1, filter_size_conv1), strides=(1, 1), padding='same')(
        inputs)
    x = ReLU()(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)

    x = Conv2D(num_filters_conv2, kernel_size=(filter_size_conv2, filter_size_conv2), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = Flatten()(x)

    print(type)

    if type == 'discrete':
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dense(num_classes)(x)
        x = predictions = Softmax()(x)

    if type == 'continuous':
        x = Dense(fc_layer_size)(x)
        x = ReLU()(x)
        x = Dense(1)(x)

    saliency_model = Model(inputs=inputs, outputs=x)
    return saliency_model


def FCNmodel(img_size, num_filters, num_channels, num_classes, type):
    inputs = Input(shape=(img_size, img_size, num_channels))

    layer_conv1_a = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    layer_conv1_b = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(layer_conv1_a)
    layer_conv2_a = Conv2D(2 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv1_b)
    layer_conv2_b = Conv2D(2 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv2_a)
    layer_conv2_c = Conv2D(2 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv2_b)
    layer_conv3_a = Conv2D(4 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv2_c)
    layer_conv3_b = Conv2D(4 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv3_a)
    layer_conv3_c = Conv2D(4 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv3_b)
    layer_conv3_d = Conv2D(4 * num_filters, (3, 3), activation='relu', padding='same')(layer_conv3_c)
    layer_upsampled_2_a = Conv2DTranspose(2 * num_filters, (3, 3), padding='same')(layer_conv3_d)
    layer_upsampled_2_a = Concatenate()([layer_conv2_a, layer_upsampled_2_a])
    layer_upsampled_2_b = Conv2D(2 * num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_2_a)
    layer_upsampled_2_c = Conv2D(2 * num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_2_b)
    layer_upsampled_1_a = Conv2DTranspose(num_filters, (3, 3), padding='same')(layer_upsampled_2_c)
    layer_upsampled_1_a = Concatenate()([layer_conv1_a, layer_upsampled_1_a])
    layer_upsampled_1_b = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_1_a)
    layer_upsampled_1_c = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_1_b)
    layer_upsampled_1_d = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_1_c)
    layer_upsampled_1_e = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(layer_upsampled_1_d)
    layer_upsampled_1_f = Conv2D(1, (3, 3), activation='relu', padding='same')(layer_upsampled_1_e)
    saliency_model = Model(inputs=inputs, outputs=layer_upsampled_1_f)
    return saliency_model
