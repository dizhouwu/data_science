def rotated_array_search(input_list, number):
    """
    Find the index by searching in a rotated sorted array

    Args:
       input_list(array), number(int): Input array to search and the number
    Returns:
       int: Index or -1
    """

    # find pivot
    left, right = 0, len(input_list)-1
        
    while left <= right:
        mid  = left + (right-left)//2
        if input_list[mid] == number:
            return mid

        elif input_list[mid] >= input_list[left]:
            if number >= input_list[left] and number < input_list[mid]:
                right = mid - 1
            else:
                left = mid+1

        else:
            if number <= input_list[right] and number > input_list[mid]:
                left = mid +1
            else:
                right = mid -1
    return -1



def linear_search(input_list, number):
    for index, element in enumerate(input_list):
        if element == number:
            return index

    return -1

def test_function(test_case):
    input_list = test_case[0]
    number = test_case[1]
    #print(linear_search(input_list, number))
    if linear_search(input_list, number) == rotated_array_search(input_list, number):
        print("Pass")
    else:
        print("Fail")

test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 6])
test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 1])
test_function([[6, 7, 8, 1, 2, 3, 4], 8])
test_function([[6, 7, 8, 1, 2, 3, 4], 1])
test_function([[6, 7, 8, 1, 2, 3, 4], 10])

# edge case as -1 == -1
test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 11])
test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 12])
test_function([[], 12])
test_function([[1], 12])