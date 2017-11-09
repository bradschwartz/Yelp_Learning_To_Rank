#!/usr/bin/env python3
import json
# from collections import OrderedDict

def load_data(file_name):
	with open(file_name) as file:
		data = [json.loads(line) for line in file]
	return data


def main():
	import sys
	file_name = sys.argv[1]
	data = load_data(file_name)
	print(data[0])
	return load_data(file_name)




if __name__ == "__main__":
	main()