import math
import random
# libraries used for the graph:
import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATE = .1
TOTAL_ITERATIONS = 100

patterns = []

# read data from a file
with open('test1.txt', 'r') as f:
	while 1:
		line = f.readline().strip().split()
		if not line:
			break
		patterns.append([float(line[0]), float(line[1]), float(line[2])])
'''
patterns[n][0] is the x coordinate
patterns[n][1] is the y coordinate
patterns[n][2] is the correct output (1 or -1)
'''

def calculate_output(weights, bias, x, y):
	output = x * weights[0] + y * weights[1] + bias
	return 1 if output >= 0 else -1


# initialize random values for weights and bias
def init_params():
	weights = [random.random() for i in range(2)]
	bias = random.random()
	return weights, bias


def update_params(weights, bias, err, x, y):
	weights[0] += LEARNING_RATE * err * x
	weights[1] += LEARNING_RATE * err * y
	bias += LEARNING_RATE * err
	return weights, bias


def graph_points():
	# assuming that there will always be a list of patterns called 'patterns'
	for pattern in patterns:
		plt.plot(pattern[0], pattern[1], 'ro' if pattern[2] == 1 else 'go')


def graph_line(x_multiplicand, y_multiplicand, bias):
	x = np.linspace(-10, 10, 100)
	plt.plot(x, (x_multiplicand*x + bias)/-y_multiplicand, '-b')


def gradient_descent():
	# initialize
	weights, bias = init_params()
	global_err = 0

	for iteration in range(TOTAL_ITERATIONS):
		global_err = 0
		# iterate over each pattern
		for pattern in patterns:
			# calculate output and local error
			output = calculate_output(weights, bias, pattern[0], pattern[1])
			local_err = pattern[2] - output
			# set new weights and bias, update global error
			weights, bias = update_params(weights, bias, local_err, pattern[0], pattern[1])
			global_err += local_err**2

		# RMSE = root mean squared error
		rmse = '{:.3f}'.format(math.sqrt(global_err/len(patterns)))
		print(f'Iteration: {iteration+1} : RMSE = {rmse}')

		# break if everything's been predicted correctly
		if global_err == 0:
			break

	'''
	we need to transform the line values to this format: y = mx + b
	0 = x_multiplicand*x + y_multiplicand*y + bias
	=>	y = (x_multiplicand*x + bias)/-y_multiplicand
	'''
	# print the first three digits of each final value
	line_equation = f'y = ({str(weights[0])[:4]}x + {str(bias)[:4]}) / -({str(weights[1])[:4]})'
	print(f'\nDecision boundary line equation: {line_equation}')

	return weights, bias


def main():
	weights, bias = gradient_descent()
	graph_points()
	graph_line(weights[0], weights[1], bias)
	plt.show()


if __name__ == '__main__':
	main()