#!/usr/bin/env python3
import sys
import load

VOCAB = set()
def get_vocab(business_string):
	global VOCAB
	for word in business_string.lower().split():
		VOCAB.add(word)


TRUE_ATTRS = ['RestaurantsPriceRange2', 'BusinessParking', 'BikeParking', 'WheelchairAccessible']
PARKING = ['garage', 'street', 'validated', 'lot', 'valet']
def parse_business_attributes(business_attributes):
	attributes = []
	for feature in TRUE_ATTRS:
		if feature != 'BusinessParking':
			attributes.append(int(business_attributes[feature]))
		else:
			for parking_feature in business_attributes[feature]:
				attributes.append(int(business_attributes[feature][parking_feature]))

	return attributes

BUS_KEYS = ['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 
			'postal_code', 'latitude', 'longitude', 'stars', 'review_count',
			 'is_open', 'attributes', 'categories', 'hours']
BUS_STR_KEYS = ['name', 'address', 'city', 'postal_code', 'state']
VAL_BUS_KEYS = [word for word in BUS_KEYS if word not in BUS_STR_KEYS]
def single_feature_vec(business):
	features = []
	for key in VAL_BUS_KEYS[1:]:
		if key != 'attributes':
			pass
		else:
			features.append(parse_business_attributes(business[key]))



def main():
	file_name = 'playdata/business.json'
	data = load.load_data(file_name)
	return data



if __name__ == "__main__":
	main()