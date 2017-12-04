#!/usr/bin/env python3
import json
from pandas.io.json import json_normalize

def load_data(file_name):
	with open(file_name) as file:
		data = json_normalize([json.loads(line) for line in file])
		# data = json_normalize([line for line in file])
	return data


def main():
	import sys
	file_name = sys.argv[1]
	data = load_data(file_name)
	print(data[0])
	print(data.head())




if __name__ == "__main__":
	main()