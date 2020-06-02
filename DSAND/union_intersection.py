class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return str(self.value)


class LinkedList:
    def __init__(self):
        self.head = None

    def __str__(self):
        cur_head = self.head
        out_string = ""
        while cur_head:
            out_string += str(cur_head.value) + " -> "
            cur_head = cur_head.next
        return out_string


    def append(self, value):

        if self.head is None:
            self.head = Node(value)
            return

        node = self.head
        while node.next:
            node = node.next

        node.next = Node(value)

    def size(self):
        size = 0
        node = self.head
        while node:
            size += 1
            node = node.next

        return size

    def to_list(self):
    	out = []
    	node = self.head
    	while node:
    		out.append(node.value)
    		node = node.next
    	return out

def union(llist_1, llist_2):
    # Your Solution Here
    list_1 = llist_1.to_list()
    list_2 = llist_2.to_list()

    all_ = list(set(list_1+list_2))

    ll = LinkedList()

    for _ in all_:
    	ll.append(_)

    return ll

def intersection(llist_1, llist_2):
    # Your Solution Here
    
    set_1 = set(linked_list_1.to_list())
    set_2 = set(linked_list_2.to_list())

    res = []

    for ele in set_1:
    	if ele in set_2:
    		res.append(ele)

    ll = LinkedList()

    for _ in res:
    	ll.append(_)
    return ll


# Test case 1

linked_list_1 = LinkedList()
linked_list_2 = LinkedList()

element_1 = [3,2,4,35,6,65,6,4,3,21]
element_2 = [6,32,4,9,6,1,11,21,1]

for i in element_1:
    linked_list_1.append(i)

for i in element_2:
    linked_list_2.append(i)

print (union(linked_list_1,linked_list_2))
print (intersection(linked_list_1,linked_list_2))

# Test case 2

linked_list_3 = LinkedList()
linked_list_4 = LinkedList()

element_1 = [3,2,4,35,6,65,6,4,3,23]
element_2 = [1,7,8,9,11,21,1]

for i in element_1:
    linked_list_3.append(i)

for i in element_2:
    linked_list_4.append(i)

print (union(linked_list_3,linked_list_4))
print (intersection(linked_list_3,linked_list_4))

# case 3: edge case
linked_list_5 = LinkedList()
linked_list_6 = LinkedList()

element_1 = []
element_2 = [1, 7, 8]

for i in element_1:
    linked_list_5.append(i)

for i in element_2:
    linked_list_6.append(i)

print(union(linked_list_5, linked_list_6))
# 8 -> 1 -> 7 ->
print(intersection(linked_list_5, linked_list_6))
#