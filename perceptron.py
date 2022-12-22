import math
import random


LEARNING_RATE = .1
TOTAL_ITERATIONS = 100

x_values = [] # x coordinates
y_values = [] # y coordinates
outputs = [] # the correct answers

# read data from a file
with open('test1.txt', 'r') as f:
	while 1:
		line = f.readline().strip().split()
		if not line:
			break
		x_values.append(float(line[0]))
		y_values.append(float(line[1]))
		if line[2] == '0':
			outputs.append(-1.0)
		else:
			outputs.append(float(line[2]))


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


def gradient_descent():
	# initialize
	weights, bias = init_params()
	global_err = 0

	for iteration in range(TOTAL_ITERATIONS):
		global_err = 0
		# iterate over each pattern
		for pattern in zip(x_values, y_values, outputs):
			# calculate output and local error
			output = calculate_output(weights, bias, pattern[0], pattern[1])
			local_err = pattern[2] - output
			# set new weights and bias, update global error
			weights, bias = update_params(weights, bias, local_err, pattern[0], pattern[1])
			global_err += local_err**2

		# RMSE = root mean squared error
		rmse = '{:.3f}'.format(math.sqrt(global_err/len(x_values)))
		print(f'Iteration: {iteration+1} : RMSE = {rmse}')

		if global_err == 0:
			break

	# print final values up to two decimal places
	print(f'\nDecision boundary line equation: {str(weights[0])[:4]}x + {str(weights[1])[:4]}y + {str(bias)[:4]} = 0')


if __name__ == '__main__':
	gradient_descent()