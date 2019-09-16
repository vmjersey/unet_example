from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose



class Unet:

    def __init__(self,shape):
        self.shape = shape

    def down(self,input_layer, filters, pool=True):

        conv1 = Conv2D(filters, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        activation='relu'
                    )(input_layer)
        residual = Conv2D(filters, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        activation='relu'
                    )(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            return max_pool, residual
        else:
            return residual

    def up(self,input_layer, residual, filters):
        filters=int(filters)
        upsample = UpSampling2D()(input_layer)
        upconv = Conv2D(filters, 
                        kernel_size=(2, 2), 
                        padding="same"
                    )(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        activation='relu'
                    )(concat)
        conv2 = Conv2D(filters, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        activation='relu'
                    )(conv1)
        return conv2



    def build_model(self):
        print(type(self.shape),self.shape)
        filters = 64
        input_layer = Input(shape=self.shape)
        layers = [input_layer]
        residuals = []


        d1, res1 = self.down(input_layer, filters)
        residuals.append(res1)
        filters *= 2

        d2, res2 = self.down(d1, filters)
        residuals.append(res2)
        filters *= 2

        d3, res3 = self.down(d2, filters)
        residuals.append(res3)
        filters *= 2

        d4, res4 = self.down(d3, filters)
        residuals.append(res4)
        filters *= 2

        d5 = self.down(d4, filters, pool=False)

        up1 = self.up(d5, residual=residuals[-1], filters=filters/2)
        filters /= 2

        up2 = self.up(up1, residual=residuals[-2], filters=filters/2)
        filters /= 2

        up3 = self.up(up2, residual=residuals[-3], filters=filters/2)
        filters /= 2

        up4 = self.up(up3, residual=residuals[-4], filters=filters/2)

        out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

        model = Model(input_layer, out)

        return model


