import numpy as np 
import predictor as p 
import argparse
import os

def buildArgParser():
    parser = argparse.ArgumentParser(description='Fast Gradient Sign Method')
    parser.add_argument('--root_dir', type=str,
                        help='directory contaning folder pgm which contains input images')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('-b', '--baseline', help='Compare with baseline', action='store_true')
    return parser

parser = buildArgParser()
args = parser.parse_args()

image_dir = os.path.join(args.root_dir,'pgm')
adv_dir = os.path.join(args.root_dir,'adv')
base_dir = os.path.join(args.root_dir,'base')

eps = args.eps

y = p.load_params('../param.txt',1024,256,23)
# print(y['W1'])


p.fgsm(args.root_dir, eps, args.baseline)

print("Original images")
p.predict(image_dir)
print()

print("Adversarial examples")
p.predict(adv_dir)
print()

if args.baseline:
	print("Baseline images with random perturbations")
	p.predict(base_dir)