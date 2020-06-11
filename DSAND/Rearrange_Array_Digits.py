def rearrange_digits(input_list):
    """
    Rearrange Array Elements so as to form two number such that their sum is maximum.

    Args:
       input_list(list): Input List
    Returns:
       (int),(int): Two maximum sums
    """
    if not input_list:
        return None

    if len(input_list) <= 1:
        return input_list

    sorted_inputs = mergesort(input_list)
    
    x = 0
    for i in range(0, len(input_list), 2):
        x = x*10+input_list[i]
    
    y = 0
    for i in range(1, len(input_list),2):
        y = y*10+input_list[i]
    
    return (x,y)
    



def mergesort(arr): 
    if len(arr) >1: 
        mid = len(arr)//2 #Finding the mid of the array 
        L = arr[:mid] # Dividing the array elements  
        R = arr[mid:] # into 2 halves 
  
        mergesort(L) # Sorting the first half 
        mergesort(R) # Sorting the second half 
  
        i = j = k = 0
          
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): 
            if L[i] > R[j]: 
                arr[k] = L[i] 
                i+=1
            else: 
                arr[k] = R[j] 
                j+=1
            k+=1
          
        # Checking if any element was left 
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1
          
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1
    return arr

def test_function(test_case):
    output = rearrange_digits(test_case[0])
    #print(output)
    solution = test_case[1]
    if sum(output) == sum(solution):
        print("Pass")
    else:
        print("Fail")


test_function([[1, 2, 3, 4, 5], [542, 31]])
test_case = [[4, 6, 2, 5, 9, 8], [964, 852]]
test_function([[4, 6, 2, 5, 9, 8], [964, 852]])
#edge case
print(rearrange_digits(None)) # None
print(rearrange_digits([123]))  # [123]