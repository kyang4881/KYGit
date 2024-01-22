from itertools import combinations as c

def find_pairs(nums, target):
    """Returns pairs with sum equal to the target
    
    Parameters
    ---------
    nums: list
    target: int
    
    Returns
    -------
    list of pairs with sum equal to the target value
    
    Example
    -------
    Input:
    
    nums = [8, 7, 2, 5, 3, 1]
    target = 10
    
    Output:
    
    [(8, 2), (7, 3)]
    
    """
    
    pairs = []
    
    for pair in c(nums, 2):
        if sum(pair) == target:
            pairs.append(pair) 
        
    return pairs
