import sys
import itertools

def to_CBST(a):
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def to_BST(root):
        """
        Convert the given root to a binary search tree
        """
        def inorder_traversal(node):
            if not node:
                return []
            return inorder_traversal(node.left) + [node] + inorder_traversal(node.right)
        
        def preorder_traversal(node):
            nonlocal index
            if not node:
                return
            preorder_traversal(node.left)
            node.key = keys[index]
            index += 1
            preorder_traversal(node.right)

        nodes = inorder_traversal(root)
        keys = [node.key for node in nodes]
        keys.sort()
        index = 0
        preorder_traversal(root)
        return root
    
    def to_tree(input_nodes, root_node):
        """
        Parse input data into a binary tree
        """
        nodes = {}
        for input_node in input_nodes:
            key, left, right = input_node[0], input_node[1], input_node[2]
            if key not in nodes:
                nodes[key] = Node(int(key))
            if left != 'x':
                if left not in nodes:
                    nodes[left] = Node(int(left))
                nodes[key].left = nodes[left]
            if right != 'x':
                if right not in nodes:
                    nodes[right] = Node(int(right))
                nodes[key].right = nodes[right]
        return nodes[root_node]
    
    def find_root(nodes):
        """
        Return a dictionary containing the root node
        """
        nodes_dict = {'root_node': []}
        for n in range(len(nodes)):
            if nodes[n][0] not in ' '.join(itertools.chain(*[node for node in nodes if node != nodes[n]])):
                nodes_dict['root_node'].append(nodes[n])
                break
        return nodes_dict
    
    def preorder_traversal(root):
        """
        Return the output of the binary search tree in preorder traversal. 
        """
        if not root:
            return ""
        return  " ".join((str(root.key) + " " + str(preorder_traversal(root.left)) + " " + str(preorder_traversal(root.right))).split()) 
     
    root_dict = find_root(a)
    parsed_tree = to_tree(input_nodes = a, root_node = root_dict['root_node'][0][0])   
    res = to_BST(parsed_tree)
    
    return preorder_traversal(res)
    
num_line = int(sys.stdin.readline())
for _ in range(num_line):
    a = [s.split(':') for s in sys.stdin.readline().split()]
    print(to_CBST(a))
