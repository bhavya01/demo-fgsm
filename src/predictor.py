
import numpy as np 
import os

def load_image_file(filename):
	with open(filename,'r') as f:
		lines = f.readlines()
		z = []
		for i in range(3,len(lines)):
			l = lines[i].strip()
			l = l.split(' ')
			for  j in range(len(l)):
				l[j] = float(l[j])
			z.append(l)

		return z

def load_params(filename, N, H, C):
	with open(filename, 'r') as f:
		lines  = f.readlines()
		params = {}
		params['W1'] = np.zeros([H,N])
		params['b1'] = np.zeros(H)
		params['W2'] = np.zeros([H,H])
		params['b2'] = np.zeros(H)
		params['W3'] = np.zeros([C,H])
		params['b3'] = np.zeros(C)
		
		index = 0
		
		for i in range(index, index+H):
			l = lines[i].strip()
			l = l.split(' ')
			for j in range(len(l)):
				params['W1'][i-index][j] = float(l[j])
		
		index = index+H

		l = lines[index].strip()
		l = l.split(' ')
		for j in range(len(l)):
			params['b1'][j] = float(l[j])

		index = index+1

		for i in range(index, index+H):
			l = lines[i].strip()
			l = l.split(' ')
			for j in range(len(l)):
				params['W2'][i-index][j] = float(l[j])
		
		index = index+H

		l = lines[index].strip()
		l = l.split(' ')
		for j in range(len(l)):
			params['b2'][j] = float(l[j])

		index = index+1

		for i in range(index, index+C):
			l = lines[i].strip()
			l = l.split(' ')
			for j in range(len(l)):
				params['W3'][i-index][j] = float(l[j])
		
		index = index+C

		l = lines[index].strip()
		l = l.split(' ')
		for j in range(len(l)):
			params['b3'][j] = float(l[j])

	return params

def forward(input,params):
	a1 = np.dot(params['W1'],input) + params['b1']
	a1[a1 < 0] = 0

	a2 = np.dot(params['W2'],a1) + params['b2']
	a2[ a2 < 0] = 0

	y = np.dot(params['W3'], a2) + params['b3']

	z = np.exp(y)
	z = z/np.sum(z)
	return z

def predict():
	image_folder = '/home/bhavya/Programming/ML_notes/intern-coding-tasks/2018/ml/pgm'
	
	correct = 0
	total = 0

	with open('../labels.txt','r') as f:
		lines = f.readlines()
		params = load_params('../param.txt', 1024, 256, 23)

		for filename in os.listdir(image_folder):
			if filename.endswith(".pgm"):
				image = load_image_file(os.path.join(image_folder,filename))
				image = np.ravel(image)/255.0

				z = forward(image,params)
				result = np.argmax(z) + 1 

				f1 = filename[0:-4]

				l = lines[int(f1)-1].strip()

				if(result == int(l)):
					correct += 1

				# print("File: {0}, predicted: {1}, actual: {2}".format(filename, result, int(l)))
				total += 1

		print("Accuracy : {0}".format(100.0*correct/total))


def fgsm():
	pass