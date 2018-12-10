import numpy as np 
import predictor as p 
x = p.load_image_file('../pgm/1.pgm')
x = np.ravel(x)
x = x/255.0
# print(x)

y = p.load_params('../param.txt',1024,256,23)
# print(y['W1'])

p.predict()