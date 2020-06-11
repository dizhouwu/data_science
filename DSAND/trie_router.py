# A RouteTrieNode will be similar to our autocomplete TrieNode... with one additional element, a handler.
class RouteTrieNode:
    def __init__(self, handler=None):
        self.children = {}
        self.handler = handler

    def insert(self, route):
        self.children[route] = RouteTrieNode()

# A RouteTrie will store our routes and their associated handlers
class RouteTrie:
    def __init__(self):
        # Initialize the trie with an root node and a handler, this is the root path or home page node
        self.root = RouteTrieNode()
        
    def insert(self, paths, handler):
        # Similar to our previous example you will want to recursively add nodes
        # Make sure you assign the handler to only the leaf (deepest) node of this path
        node = self.root
        
        for path in paths:
            if path not in node.children:
                node.children[path] = RouteTrieNode()
            node = node.children[path]
        
        node.handler = handler
        
    def find(self, paths):
        # Starting at the root, navigate the Trie to find a match for this path
        # Return the handler for a match, or None for no match
        node = self.root
        
        for path in paths:
            if path not in node.children:
                return None
            node = node.children[path]
        
        return node.handler

class Router:
    def __init__(self, handler, not_found_handler="404"):
        self.routes = RouteTrie()
        self.routes.insert("/", handler)
        self.not_found_handler = not_found_handler
    
    def add_handler(self, route, handler):
        paths = self._split_path(route)
        self.routes.insert(paths, handler)
    

    def lookup(self, route):
        paths = self._split_path(route)
        return self.routes.find(paths) or self.not_found_handler
    
    
    def _split_path(self, route):
        if len(route)==1:
            return ['/']
        else:
            return route.strip('/').split('/')
    

# create the router and add a route
router = Router("root handler", "not found handler") # remove the 'not found handler' if you did not implement this
router.add_handler("/home/about", "about handler")  # add a route

# some lookups with the expected output
print(router.lookup("/")) # should print 'root handler'
print(router.lookup("/home")) # should print 'not found handler' or None if you did not implement one
print(router.lookup("/home/about")) # should print 'about handler'
print(router.lookup("/home/about/")) # should print 'about handler' or None if you did not handle trailing slashes
print(router.lookup("/home/about/me")) # should print 'not found handler' or None if you did not implement one
