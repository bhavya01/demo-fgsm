import matplotlib.pyplot as plt
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

def write_image_file(image, filename):
	np.savetxt(filename, image, fmt='%d', delimiter=' ', newline='\n', header='P2\n32 32\n255', comments='')


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
	relu1 = [a1 < 0]
	a1[a1 < 0] = 0

	a2 = np.dot(params['W2'],a1) + params['b2']
	relu2 = [a2 < 0]
	a2[ a2 < 0] = 0

	y = np.dot(params['W3'], a2) + params['b3']

	z = np.exp(y)
	z = z/np.sum(z)

	cache = (relu1,relu2)

	return z, cache

def backward(probs, label, params, cache):
	relu1 = cache[0]
	relu2 = cache[1]

	dy = probs
	dy[label] -= 1

	dh2 = np.dot(params['W3'].T, dy)

	da2 = dh2
	da2[relu2] = 0

	dh1 = np.dot(params['W2'].T, da2)

	da1 = dh1
	da1[relu1] = 0

	dx = np.dot(params['W1'].T, da1)

	return dx

def predict(image_folder):
	
	correct = 0
	total = 0

	with open('../labels.txt','r') as f:
		lines = f.readlines()
		params = load_params('../param.txt', 1024, 256, 23)

		for filename in os.listdir(image_folder):
			if filename.endswith(".pgm"):
				image = load_image_file(os.path.join(image_folder,filename))
				image = np.ravel(image)/255.0

				z, _ = forward(image,params)
				result = np.argmax(z) + 1 

				f1 = filename[0:-4]

				l = lines[int(f1)-1].strip()

				if(result == int(l)):
					correct += 1

				# print("File: {0}, predicted: {1}, actual: {2}".format(filename, result, int(l)))
				total += 1

		print("Accuracy : {0:0.4f}".format(100.0*correct/total))

def normalize_and_reshape(input):
	a = np.min(input)
	b = np.max(input)

	out = 255.0*(input - a)/(b - a)
	out.astype(np.uint8)
	out = np.reshape(out,[32,32])

	return out

def fgsm(root_dir, eps = 0.1, generate_baseline=False):

	image_folder = os.path.join(root_dir,'pgm')
	output_folder = os.path.join(root_dir,'adv')

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	baseline_folder = os.path.join(root_dir,'base')
	if generate_baseline and not os.path.exists(baseline_folder):
		os.makedirs(baseline_folder)

	with open('../labels.txt','r') as f:
		lines = f.readlines()
		params = load_params('../param.txt', 1024, 256, 23)

		for filename in os.listdir(image_folder):
			if filename.endswith(".pgm"):
				image = load_image_file(os.path.join(image_folder,filename))
				image = np.ravel(image)/255.0

				z, cache = forward(image,params)

				relu1 = cache[0]
				relu2 = cache[1]

				# t is the actual label
				f1 = filename[0:-4]
				t = lines[int(f1)-1].strip()
				t = int(t) - 1

				dx = backward(z, t, params, cache)

				adv = image + eps * np.sign(dx)

				adv_norm = normalize_and_reshape(adv)

				write_image_file(adv_norm, os.path.join(output_folder, filename))

				if generate_baseline:
					base = image + eps * np.sign(np.random.randn(1024))
					base_norm = normalize_and_reshape(base)
					write_image_file(base_norm, os.path.join(baseline_folder, filename))