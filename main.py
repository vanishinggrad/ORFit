import scipy as sp
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=4)

def rotate(xs):
    # rotate elements along axis 0 by random angles in [0, 180]

    angles = (180 * np.random.random(size=xs.shape[0]).astype('float32'))
    xs_new = []
    for i, (x, angle) in enumerate(zip(xs, angles)):
        x_rotated = sp.ndimage.rotate(x, angle=angles[i], reshape=False, mode='nearest')
        xs_new.append(x_rotated)
    xs_new = np.array(xs_new)
    return xs_new, angles

def gen_mnist_regression_data(digit=2):
    # get digit-specific MNIST data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    (x_train_rotated, y_train), (x_test_rotated, y_test) = (rotate(x_train[y_train == digit]), rotate(x_test[y_test == digit]))
    return (x_train_rotated, y_train), (x_test_rotated, y_test)



def get_model():
    # a 'linear' model in TensorFlow

    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1, activation='relu')])

    optimizer = tf.keras.optimizers.Adam(0.01) # not needed, as the optimization is carried out manually
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)

    print(model.summary())
    return model, loss_fn

def get_stoch_gradient(model, loss_fn, inputs, labels):
    # stochastic gradient with respect to a single (input, output pair), note usage in the code below

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        pred_loss = loss_fn(labels, predictions)
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    return np.hstack([arr.numpy().ravel() for arr in gradients])

def get_gradient(model, inputs):
    # gradient with respect to a model output, refered to as \del f_i in the paper

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
    gradients = tape.gradient(predictions, model.trainable_variables)
    return np.hstack([arr.numpy().ravel() for arr in gradients])

def get_projection(basis, vec):
    # project a vector onto a given basis

    if not (len(basis) > 0):
        return np.zeros_like(vec)

    vec_sum = np.zeros_like(vec)
    for col in range(basis.shape[1]):
        col = basis[:, col].ravel()
        vec_projected = np.dot(vec, col) * col / (np.linalg.norm(col, ord=2) ** 2 + 1e-10)
        vec_sum += vec_projected

    return vec_sum

def sample_data_sequence(inputs, labels, n=100):
    # the authors say that "an image rotated with a smaller angle arrives earlier"
    # below is a simple weighing scheme based on the reciprocal of probability

    idx_argsort = np.random.choice(np.arange(len(labels)), size=n,
                                   replace=False, p=(1 / labels) / np.sum(1 / labels))
    inputs, labels = inputs[idx_argsort], labels[idx_argsort]
    return inputs, labels

def set_model_weights(vec, model):

    # set model weights

    start = 0
    end = None
    for tv in model.trainable_variables:
        size = tf.math.reduce_prod(tv.shape)
        end = start + size
        tv.assign(tf.reshape(vec[start:end].astype('float32'), shape=tv.shape))
        start = end

def eval_model(model, x_train, y_train, x_test, y_test, return_arr = False, tf_verbose=False):

    eval_train, eval_test = (np.sqrt(model.evaluate(x_train, y_train, verbose=tf_verbose)),
                             np.sqrt(model.evaluate(x_test, y_test, verbose=tf_verbose)))
    print("RMSE (deg)", "\t", f"Train: {eval_train}", f"Test: {eval_test}")
    if return_arr:
        return eval_train, eval_test




def ORfit(ds_size = 100, m=10, digit = 2):

    n_ds = ds_size # number of elements in the data sequence
    model, loss_func = get_model()
    (x_train, y_train), (x_test, y_test) = gen_mnist_regression_data(digit=digit)
    x_train, y_train = sample_data_sequence(x_train, y_train, n=n_ds)

    logs = [eval_model(model, x_train, y_train, x_test, y_test, return_arr=True)]

    U = np.array([])
    E = np.array([])
    w = np.hstack([arr.numpy().ravel() for arr in model.trainable_variables])
    m = m

    for i in range(n_ds):

        g = get_stoch_gradient(model, loss_func, np.expand_dims(x_train[i], axis=0), y_train[i])
        g_prime = g - get_projection(U, g)
        v_prime = (get_gradient(model, np.expand_dims(x_train[i], axis=0)) \
                   - get_projection(U, get_gradient(model, np.expand_dims(x_train[i], axis=0))))

        if i <= (m-1):
            if i == 0:
                U = np.expand_dims(v_prime, axis=1)
            else:
                U = np.append(U, np.expand_dims(v_prime, axis=1), axis=1)

            if i == m-1:
                temp_shape = U.shape
                U, E_vals, V_temp = sp.linalg.svd(U)
                E = sp.linalg.diagsvd(E_vals, *temp_shape)


        else:
            u = v_prime / (1e-10 + np.linalg.norm(v_prime, ord=2))

            temp = E
            temp = np.append(temp, np.zeros(shape=(temp.shape[0], 1)), axis=1)
            temp = np.append(temp, np.zeros(shape=(1, temp.shape[1])), axis=0)
            temp[-1, -1] = np.dot(u, v_prime)

            U_prime, E_vals, V_temp = sp.linalg.svd(temp)
            E = sp.linalg.diagsvd(E_vals, *temp.shape)

            U = np.append(U, np.expand_dims(u, axis=1), axis=1)
            U = np.matmul(U, U_prime)
            U, E = U[:, :m], E[:m, :]

        eta = (tf.squeeze(model(np.expand_dims(x_train[i], 0)) - y_train[i]).numpy()) / (1e-10 + np.dot(get_gradient(model, np.expand_dims(x_train[i], 0)),
                                                                                                        g_prime))

        w = w - eta * g_prime

        set_model_weights(w, model)


        if i == 0 or i % 10 == 0:
            eval_model(model, x_train, y_train, x_test, y_test)
        logs.append(eval_model(model, x_train, y_train, x_test, y_test, return_arr=True))

    return logs


ORfit(100, m=10, digit=2)