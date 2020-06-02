import hashlib
from datetime import datetime
class Block:

    def __init__(self, timestamp, data, previous_hash):
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calc_hash(data)
        self.next = None

    def calc_hash(self, data: str):
        sha = hashlib.sha256()

        hash_str = data.encode('utf-8')

        sha.update(hash_str)

        return sha.hexdigest()
    
class BlockChain:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, value):
        if not value:
            return 
        
        self.size += 1
        node = self.head
        
        if not node:
            block = Block(datetime.now(), value, None)
            self.head = block
        else:
            while node.next:
                node = node.next
            node.next = Block(datetime.now(), value, node.hash)
            
            
# test
chain = BlockChain()
for i in 'kono dio da'.split():
    chain.append(i)
chain.append('data for a')
chain.append('data for b')
chain.append('data for c')

print(chain.head.data)
# kono
print(chain.head.next.hash == chain.head.next.next.previous_hash)
# True