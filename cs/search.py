def binary_search(array, item):

    start = 0
    end = len(array) - 1

    mid_point = (start + end) // 2

    if item == array[mid_point]:
        return mid_point

    elif item < array[mid_point]:
        # slicing is O(n) so this implementation is not most efficient
        new_array = array[start:mid_point]
        return binary_search(new_array, item)

    else:
        new_array = array[mid_point + 1:end + 1]
        return mid_point + binary_search(new_array, item) + 1
