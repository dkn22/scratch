def bubblesort(array):

    # for i in range(len(array) - 1, 0, -1):
    #     for j in range(i):
    #         if array[j] > array[j + 1]:
    #             array[j], array[j + 1] = array[j + 1], array[j]

    i = len(array) - 1
    swapped = True

    # early stopping if no swaps (list is sorted)
    while i > 0 and swapped:
        swapped = False

        for j in range(i):
            if array[j] > array[j + 1]:
                swapped = True
            array[j], array[j + 1] = array[j + 1], array[j]

        i -= 1

    return array


def selectionsort(array):

    for i in range(len(array) - 1, 0, -1):
        argmax = 0

        for j in range(1, i + 1):
            if array[j] > array[argmax]:
                argmax = j

        array[i], array[argmax] = array[argmax], array[i]

    return array


def insertionsort(array):

    # for i in range(1, len(array)):  # n-1 passes
    #     idx = i
    #     current_value = array[idx]

    #     while idx > 0 and array[idx - 1] > current_value:
    #     	# shift larger values to the right
    #         array[idx] = array[idx - 1]
    #         idx -= 1

    #     # "insert" the value at hand into the appropriate slot
    #     array[idx] = current_value

    sublist = [array[0]]
    for i in range(1, len(array)):
        idx = i
        current_value = array[idx]
        while idx > 0 and sublist[idx - 1] > current_value:
            idx -= 1

        sublist.insert(idx, current_value)

    return sublist


def shellsort(array):

    # gap is multiples of two here, but can be more efficient
    gap = len(array) // 2

    while gap > 0:

        for start in range(gap):
            # insertion sort for each sublist
            for j in range(start + gap, len(array), gap):

                current_value = array[j]
                idx = j

                while idx >= gap and array[idx - gap] > current_value:
                    array[idx] = array[idx - gap]
                    idx = idx - gap

                array[idx] = current_value

        gap = gap // 2

    return array


def mergesort(array):

    if len(array) <= 1:  # base case
        return array

    else:
        midpoint = len(array) // 2
        left_half = mergesort(array[:midpoint])
        right_half = mergesort(array[midpoint:])

        # merging
        i, j, k = 0, 0, 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                array[k] = left_half[i]
                i += 1
            else:
                array[k] = right_half[j]
                j += 1

            k += 1

        while i < len(left_half):
            array[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            array[k] = right_half[j]
            j += 1
            k += 1

    return array


def quicksort(array, start=0, end=None):

    if end is None:
        end = len(array) - 1

    # if len(array[start:end + 1]) <= 1:
    #     return array

    if start < end:

        pivot = array[start]
        leftmark = start + 1
        rightmark = end

        while rightmark >= leftmark:
                # the partition process
            while leftmark <= rightmark and array[leftmark] <= pivot:
                leftmark += 1

            while rightmark >= leftmark and array[rightmark] >= pivot:
                rightmark -= 1

            if rightmark < leftmark:  # found the split point
                break

            # values to the left of split point (and pivot) will be smaller to the right
            array[leftmark], array[rightmark] = array[rightmark], array[leftmark]

        # put pivot into split point
        splitpoint = rightmark
        array[start], array[splitpoint] = array[splitpoint], pivot

        # avoid slicing by passing indices
        array = quicksort(array, start=start, end=splitpoint - 1)
        array = quicksort(array, start=splitpoint + 1, end=end)

    return array
