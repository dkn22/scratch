import string
import operator
from collections import Counter


def is_anagram(left, right):
    """Anagram detection in O(n)"""

    assert isinstance(left, str) and isinstance(right, str)

    char_dict = {k: 0 for k in set(string.printable)}

    left_count = Counter(char_dict)
    right_count = Counter(char_dict)

    try:
        for i in left:
            left_count[i] += 1

        for j in right:
            right_count[j] += 1

    except KeyError:
        raise KeyError('Invalid character present in string.')

    return left_count == right_count


def is_rotation(left, right):
    """Check if left item is a rotation of the right"""

    concat = left + left

    return len(left) == len(right) and right in concat


def longest_subpalindrome_slice(text):
    """Manacher algorithm approach - O(n)"""

    # pad with zeros to have an odd number of chars
    # concat slashes at start and end to avoid boundary checks
    text_with_spaces = ' '.join('\{}/'.format(text.lower()))

    # array containing the max length of a palindrome centered at i
    n = len(text_with_spaces)
    palindrome_lengths = [0] * n

    center = 0
    right_edge = 0
    dist_to_edge = 0

    for i in range(1, n - 1):  # i = pRight

        # multiplication by 2 because of space padding
        pleft = 2 * center - i

        # The palidrome centered at pLeft does not exceed the right boundary
        # then it must be that the reflected palindrome at pRight has same length
        if palindrome_lengths[pleft] <= dist_to_edge and right_edge - i >= pleft:

            palindrome_lengths[i] = int(palindrome_lengths[pleft])

        else:

            # get palindrom length centered at i
            # because we padded the string with spaces
            # between each pair of characters
            # palindrome length = distance btw center and edge
            while text_with_spaces[i + palindrome_lengths[i] + 1] \
                    == text_with_spaces[i - palindrome_lengths[i] - 1]:
                palindrome_lengths[i] += 1

            if palindrome_lengths[i] > 1:
                center = i
                right_edge = center + palindrome_lengths[i]

        dist_to_edge = right_edge - i

    center, longest_palindrome = max(enumerate(palindrome_lengths),
                                     key=operator.itemgetter(1))

    return text[(center - longest_palindrome) // 2:
                (center + longest_palindrome) // 2]
