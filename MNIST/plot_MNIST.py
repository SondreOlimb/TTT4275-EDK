import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




def display_image(images,number_to_plot):

    for count,images in enumerate(images):
        test_image = np.reshape(images[2],(28,28))
        pred_image = np.reshape(images[3], (28, 28))
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
        ax1.imshow(test_image, interpolation='nearest')
        ax1.set_title(images[0][0])
        ax2.imshow(pred_image, interpolation='nearest')
        ax2.set_title(images[1][0])
        title = 'Real number:'+str(images[0][0])
        fig.suptitle(title)
        plt.draw()


        if(count == number_to_plot):
            return 0

