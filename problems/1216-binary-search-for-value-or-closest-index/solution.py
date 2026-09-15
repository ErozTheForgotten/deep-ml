def binary_search_closest(arr, target):
    # arr is sorted in non-decreasing order
    # return an integer index (or -1 for empty arr)
    n = len(arr)

    if not arr:
        return -1
    
    closest = 0
    
    for x in range(n):
        dist_from_t = abs(target - arr[x])
        
        if dist_from_t < abs(target - arr[closest]):
            closest = x
        
    return closest
       
