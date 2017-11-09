#!/usr/bin/env python3
import load
from string import punctuation

def remove_punc(word_string):
	for punc in punctuation:
		word_string = word_string.replace(punc, ' ')
	return word_string
	# return word_string.translate(str.maketrans('','',punctuation))

def extract_vocab_data(data):
	vocab = set()
	for row in data:
		for key in row:
			val = row[key]
			# add strings
			if isinstance(val, str) and key[-3:] != '_id':
				vocab.update(remove_punc(val.strip()).lower().split())
			# recursively add dictionaries
			elif isinstance(val, dict):
				vocab.update(extract_vocab_data([val]))
			# otherwise do nothing
			else:
				continue
	return vocab

def extract_vocab_file(file_name):
	data = load.load_data(file_name)
	vocab = extract_vocab_data(data)
	return vocab

def save_vocab(vocab_file, vocab):
	with open(vocab_file, 'r') as file:
		file_vocab = {word.strip() for word in file}

	file_vocab.update(vocab)
	with open(vocab_file, 'w') as file:
		file.write(' \n'.join(str(word) for word in sorted(file_vocab)))

def main(file_name, vocab_file):
	print(file_name)
	vocab = extract_vocab_file(file_name)
	# print(vocab, len(vocab), type(vocab))
	save_vocab(vocab_file, vocab)

if __name__ == "__main__":
	import sys
	if len(sys.argv) != 3:
		raise Exception("Wrong number of arguments, only path of file name and path of vocab.")
	
	main(sys.argv[1], sys.argv[2])