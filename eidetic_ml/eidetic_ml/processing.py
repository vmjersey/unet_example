import numpy as np

class Encoding:

    """The Encoding Class contains functions for encoding and decoding images.
    Methods
    -----
    decode_rle_mask(encoded_pixels,height,width)
        Takes in an rle encoded mask and returns the decoded mask.
    
    """


    def decode_rle_mask(rle_string,height,width):
        """Takes in an rle encoded mask and returns the mask.
        
        Parameters
        ----------
        encoded_pixels : str, required
            A string containing the run length encoded mask
        height: int, required
        width: int, required

        """    

    
        if rle_string == None:
            return np.zeros((height, width))
        else:
            rle_numbers = []
            for num_string in rle_string.split(' '):
                rle_numbers.append(int(num_string))

            rle_pairs = np.array(rle_numbers).reshape(-1,2)
            img = np.zeros(height*width, dtype=np.uint8)
            for index, length in rle_pairs:
                index -= 1
                img[index:index+length] = 255
            img = img.reshape(width,height)
            img = img.T
            mask = np.stack((img,)*3, axis=-1)
            return mask



