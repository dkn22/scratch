import functools


def factorial(n):
    if n <= 0:
        return 1

    return n * factorial(n - 1)


def conv_n_to_str(n, base):
    chars = '0123456789ABCDEF'
    if n < base:
        return chars[n]
    else:
        return conv_n_to_str(n // base, base) + chars[n % base]


def reverse_string(string):
    if len(string) == 1:
        return string
    return string[-1] + reverse_string(string[:-1])


def is_palindrome(s):
    import string
    table = str.maketrans({key: None for key in string.punctuation})
    clean_s = s.translate(table).replace(" ", "")

    if len(clean_s) <= 1:
        return True

    else:
        return clean_s[0] == clean_s[-1] and is_palindrome(clean_s[1:-1])


def tower_of_hanoi(height, first_pole, final_pole, intermediate_pole):
        # base case: height = 0 - do nothing
    def move_disk(first, final):
        print('Move disk from %s to %s.' % (first, final))
    if height > 0:
        # move a tower of height-1 to intermediate pole
        tower_of_hanoi(height - 1, first_pole, intermediate_pole, final_pole)
        # move remaining disk from first pole to final pole
        move_disk(first_pole, final_pole)
        # move a tower of height-1 from intermediate to final pole
        tower_of_hanoi(height - 1, intermediate_pole, final_pole, first_pole)


@memo_changemaker
def changemaker(coinlist, total_change):
    if total_change in coinlist:  # base case
        return 1

    min_n = total_change
    valid_coins = [coin for coin in coinlist if coin < total_change]

    for i in valid_coins:
        n = 1 + changemaker(valid_coins, total_change - i)
        if n < min_n:
            min_n = n

    return min_n


def dp_changemaker(coinlist, total_change, show_change=True):
    # if total_change in coinlist:  # base case
    #     return 1

    solutions = {}  # min number of coins required
    last_coins = {}  # last coin used in change

    # solve every small sub-problem on the way to the main problem
    for cents in range(total_change + 1):
        min_n = cents
        last_coin = 1
        valid_coins = [coin for coin in coinlist if coin <= cents]

        for i in valid_coins:
            solution = solutions.get(cents - i, 0) + 1

            if solution < min_n:
                min_n = solution
                last_coin = i

        solutions[cents] = min_n
        last_coins[cents] = last_coin

    if show_change:
        remaining = total_change
        print('Change for %s requires coins:' % remaining)
        while remaining > 0:
            last = last_coins[remaining]
            print(last)
            remaining = remaining - last

    return solutions[total_change]


def memo_changemaker(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}

    @functools.wraps(f)
    def _f(*args):
        try:
            return cache[args[1]]
        except KeyError:
            cache[args[1]] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(*args)
    return _f
