import sys, heapq

def get_freq(msg):
	# create a char-freq mapping
	chars = dict()
	for char in msg:
		if chars.get(char):
			chars[char] +=1
		else:
			chars[char] = 1
	return chars.items()

def make_tree(freqs):

	# freq: list(tuple(int, str))

	heap = []
	for freq in freqs:
		heapq.heappush(heap, [freq])

	while len(heap)>1:
		left = heapq.heappop(heap)
		right = heapq.heappop(heap)
		freq_0, label_0 = left[0]
		freq_1, label_1 = right[0]
		node = [(freq_0 + freq_1, label_0+label_1), left, right]
		heapq.heappush(heap, node)
	return heap.pop()

def make_map(tree, map = dict(), prefix=''):
	# could also initialize global dict
	if len(tree) == 1:
		label, freq = tree[0]
		map[label] = prefix
	else:
		value, left, right = tree
		make_map(left, map, prefix+'0')
		make_map(right, map, prefix+'1')
	return map

def huffman_encoding(msg):
	tree = make_tree(get_freq(msg))
	map = make_map(tree)
	res = ''.join([map[letter] for letter in msg])
	return res, tree

def huffman_decoding(data, tree):
	tmp_tree = tree
	res = []

	for d in data:
		if d == '0':
			tmp_tree = tmp_tree[1]
		else:
			tmp_tree = tmp_tree[2]

		if len(tmp_tree)==1:
			label, freq = tmp_tree[0]
			res.append(label)
			tmp_tree = tree

	return ''.join(res)


if __name__ == "__main__":
    codes = {}

    a_great_sentence = "The bird is the word"

    print ("The size of the data is: {}\n".format(sys.getsizeof(a_great_sentence)))
    print ("The content of the data is: {}\n".format(a_great_sentence))

    encoded_data, tree = huffman_encoding(a_great_sentence)

    print ("The size of the encoded data is: {}\n".format(sys.getsizeof(int(encoded_data, base=2))))
    print ("The content of the encoded data is: {}\n".format(encoded_data))

    decoded_data = huffman_decoding(encoded_data, tree)

    print ("The size of the decoded data is: {}\n".format(sys.getsizeof(decoded_data)))
    print ("The content of the encoded data is: {}\n".format(decoded_data))


    print("----------------------------------------------")
    a_long_sentence = "kono dio da"
    print ("The content of the data is: {}\n".format(a_long_sentence))
    print ("The size of the data is: {}\n".format(sys.getsizeof(a_long_sentence)))
    encoded_data, tree = huffman_encoding(a_long_sentence)
    print ("The size of the encoded data is: {}\n".format(sys.getsizeof(int(encoded_data, base=2))))
    print ("The content of the encoded data is: {}\n".format(encoded_data))
    decoded_data = huffman_decoding(encoded_data, tree)
    print ("The content of the encoded data is: {}\n".format(decoded_data))
    print ("The size of the decoded data is: {}\n".format(sys.getsizeof(decoded_data)))

    print("----------------------------------------------")
    a_long_sentence = "yakamashii"
    print ("The content of the data is: {}\n".format(a_long_sentence))
    print ("The size of the data is: {}\n".format(sys.getsizeof(a_long_sentence)))
    encoded_data, tree = huffman_encoding(a_long_sentence)
    print ("The size of the encoded data is: {}\n".format(sys.getsizeof(int(encoded_data, base=2))))
    print ("The content of the encoded data is: {}\n".format(encoded_data))
    decoded_data = huffman_decoding(encoded_data, tree)
    print ("The content of the encoded data is: {}\n".format(decoded_data))
    print ("The size of the decoded data is: {}\n".format(sys.getsizeof(decoded_data)))