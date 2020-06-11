import os

for epoch in range(60):
	if (epoch // 10) % 3 == 0:
	    input_size = 256
	elif (epoch // 10) % 3 == 1:
	    input_size = 380
	elif (epoch // 10) % 3 == 2:
	    input_size = 456
	print(input_size)