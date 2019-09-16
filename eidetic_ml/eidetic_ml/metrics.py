from keras import backend as K


class dice:

    def dice_coef(y_true, y_pred, smooth=1):
        """ source: 
        https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html#dice_loss
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return answer

