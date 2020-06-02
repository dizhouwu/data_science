"""
Read file into texts and calls.
It's ok if you don't understand how to read files
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
TASK 2: Which telephone number spent the longest time on the phone
during the period? Don't forget that time spent answering a call is
also time spent on the phone.
Print a message:
"<telephone number> spent the longest time, <total time> seconds, on the phone during 
September 2016.".
"""

if __name__ == '__main__':

	time = dict()
	for call in calls:
		if call[0] in time:
			time[call[0]] += int(call[3])
		else:
			time[call[0]] = int(call[3])
		if call[1] in time:
			time[call[1]] += int(call[3])
		else:
			time[call[1]] = int(call[3])

	max_t = 0
	max_num = ''

	for t in time:
		if time[t] >= max_t:
			max_t = time[t]
			max_num = t

	print("{} spent the longest time, {} seconds, on the phone during September 2016.".format(
	max_num, max_t))