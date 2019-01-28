import tensorflow as tf

l = tf.keras.layers

def _conv_block(inputs: tf.Tensor, filters: int, repeat=2) -> tf.Tensor:
    """Repeated 3x3 2d convolutions with ReLU"""
    layer = inputs
    for _ in range(repeat):
        layer = l.Conv2D(filters, kernel_size=3, activation=tf.nn.relu, padding='same')(layer)
    return layer


def _up_conv(inputs: tf.Tensor) -> tf.Tensor:
    """Deconvolution"""
    input_shape = inputs.get_shape().as_list()
    size = [2*input_shape[1], 2*input_shape[2]]
    # Try nearest neighbour resize method 
    # https://medium.com/towards-data-science/autoencoders-introduction-and-implementation-3f40483b0a85
    return tf.image.resize_images(inputs, size=size)


def unet_model(features, labels, mode, params):
    """Function to build the unet model architecture"""

    # Provide sample of original image to tensorboard
    tf.summary.image('inputs', features['image'])
    
    # Even though we have two classes we are actually only predicting one
    # The other is given by its abscence and hence binary classification
    NUM_CLASSES = 1
    
    if labels is not None:
        # Sample of our truth label for tensorboard
        tf.summary.image('labels', labels)

    # Use the image as the input or our current head of the model
    net = features['image']
    
    # Used to store layers that are copied to the up-path of the unet model
    copy_layers = []
    # Define the number of filters in each stage of the up and down path
    filter_counts = [65, 128, 256, 512]

    # Contractive path
    for num_filters in filter_counts:
        net = _conv_block(net, num_filters)
        copy_layers.append(net)
        net = l.MaxPool2D(pool_size=2, strides=2, padding='same')(net)
    
    # Apply final conv that doesn't have output copied and no maxpool
    net = _conv_block(net, 2 * filter_counts[-1])

    # Expansive path
    for num_filters in reversed(filter_counts):
        net = _up_conv(net)
        copy_layer = copy_layers.pop()
        net = l.concatenate([net, copy_layer], axis=3)
        net = _conv_block(net, num_filters)

    # Conv 1x1 to get output segmentation map
    logits = l.Conv2D(filters=NUM_CLASSES, kernel_size=1, activation=None, padding='same')(net)


    if NUM_CLASSES < 2:
        head = tf.contrib.estimator.binary_classification_head()

        # provide summary for predicted output
        predictions = tf.math.round(tf.math.sigmoid(logits))
        tf.summary.image('predictions', predictions)
    else:
        head = tf.contrib.estimator.multi_class_head(n_classes=NUM_CLASSES)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits)

