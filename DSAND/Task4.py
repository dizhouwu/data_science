"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
TASK 4:
The telephone company want to identify numbers that might be doing
telephone marketing. Create a set of possible telemarketers:
these are numbers that make outgoing calls but never send texts,
receive texts or receive incoming calls.

Print a message:
"These numbers could be telemarketers: "
<list of numbers>
The list of numbers should be print out one per line in lexicographic order with no duplicates.
"""

if __name__ == '__main__':
	call_senders = set([data[0] for data in calls])
	call_receivers = set([data[1] for data in calls])
	msg_senders = set([data[0] for data in texts])
	msg_receivers = set([data[1] for data in texts])

	candidates = []

	for sender in call_senders:
		if sender not in call_receivers and sender not in msg_senders and sender not in msg_receivers:
			candidates.append(sender)

	candidates.sort()
	print("\n These numbers could be telemarketers:")
	for cand in candidates:
		print(cand)