from keras.losses import binary_crossentropy
import keras.backend as K


def dice(y_true, y_pred):
    TP = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return 2.0 * TP / union if union != 0 else 0


def dice_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred)


def get_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


