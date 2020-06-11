def get_min_max(ints):
    """
    Return a tuple(min, max) out of list of unsorted integers.

    Args:
       ints(list): list of integers containing one or more integers
    """
	
    min_ = ints[0]
    max_ = ints[0]
    for num in ints:
        if num > max_:
            max_ = num
        if num < min_:
            min_ = num

    return (min_, max_)

## Example Test Case of Ten Integers
import random

l = [i for i in range(0, 10)]  # a list containing 0 - 9
random.shuffle(l)

print ("Pass" if ((0, 9) == get_min_max(l)) else "Fail")
print(get_min_max([1,2,3])) # (1,3)
# edge
print(get_min_max([1])) # (1,1)
# no edge case for ints requires as the problem states
# "Args:
#       ints(list): list of integers containing one or more integers


