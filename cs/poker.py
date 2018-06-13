"""Written as part of the Design of Computer Programs course by Peter Norvig"""

import random
import itertools
from collections import Counter


mydeck = [r + s for r in '23456789TJQKA' for s in 'SHDC']


def deal(numhands, n=5, deck=mydeck):

    random.shuffle(deck)

    return [deck[n * i:n * (i + 1)] for i in range(numhands)]


def poker(hands):
    '''Return the best hand: poker([hand, ...]) => [best_hand, ...]'''

    # return max(hands, key=abs)
    # return max(hands, key=hand_rank)

    return allmax(hands, func=hand_rank)


def allmax(iterable, func=None):
    """Return a list of all items equal to the max of the iterable."""

    # sorted_items = sorted(iterable, key=key, reverse=True)

    # return [x for x in sorted_items if key(x) == key(sorted_items[0])]

    func = func or (lambda x: x)

    max_item = max(iterable, key=func)
    result = list(filter(lambda i: func(i) == func(max_item), iterable))

    return result


def hand_rank(hand):
    '''
    There are nine different types of hands.

    But assigning an integer value to type of hand
    is insufficient to assign winners. E.g., pair of 10s vs of 9s.
    We need to also rank the same types of hands.

    # if straight(ranks) and flush(hand):            # straight flush
    #     return (8, max(ranks))
    # elif kind(4, ranks):                           # 4 of a kind
    #     return (7, kind(4, ranks), kind(1, ranks))
    # elif kind(3, ranks) and kind(2, ranks):        # full house
    #     return (6, kind(3, ranks), kind(2, ranks))
    # elif flush(hand):                              # flush
    #     return (5, ranks)
    # elif straight(ranks):                          # straight
    #     return (4, max(ranks))
    # elif kind(3, ranks):                           # 3 of a kind
    #     return (3, kind(3, ranks), ranks)
    # elif two_pair(ranks):                          # 2 pair
    #     return (2, two_pair(ranks), ranks)
    # elif kind(2, ranks):                           # kind
    #     return (1, kind(2, ranks), ranks)
    # else:                                          # high card
    #     return (0, ranks)

    '''

    # use tuples for lexicographic ordering
    ranks = card_ranks(hand)
    counts = tuple(sorted(Counter(ranks).values(), reverse=True))

    straight = sum([x - ranks[-1] for x in ranks]
                   ) == 10 and len(set(ranks)) == 5
    flush = len(set([s for r, s in hand])) == 1

    rank_map = {
        (5,): 10,
        (4, 1): 7,
        (3, 2): 6,
        (3, 1, 1): 3,
        (2, 2, 1): 2,
        (2, 1, 1, 1): 1,
        (1, 1, 1, 1, 1): 0
    }

    return (max(rank_map[counts], 4 * straight + 5 * flush), ranks)


def card_ranks(cards):
    '''Return a list of the ranks, sorted with higher first.'''
    ranks = [r for r, s in cards]
    names_to_ranks = {
        'T': 10,
        'J': 11,
        'Q': 12,
        'K': 13,
        'A': 14
    }

    ranks = [names_to_ranks[x] if x in names_to_ranks.keys() else int(x)
             for x in ranks]

    # alternative
    # ranks = ['--23456789TJQKA'.index(r) for r,s in hands]

    ranks.sort(reverse=True)

    # special case: Ace-low straght
    if ranks == [14, 5, 4, 3, 2]:
        ranks = [5, 4, 3, 2, 1]

    return ranks


def straight(ranks):
    '''Return True if the ordered ranks form a 5-card straight.'''
    # sum of differences from the lowest number
    # sum_diff = sum([x - ranks[-1] for x in ranks]) # this should be 10

    sum_diff = sum([x - ranks[-1] for x in ranks])
    return sum_diff == 10 and (max(ranks) - min(ranks)) == 4 and len(set(ranks)) == 5


def flush(hand):
    '''Return True if all the cards have the same suit.'''
    suits = [s for r, s in hand]
    return len(set(suits)) == 1


def kind(n, ranks):
    """Return the first rank that this hand has exactly n of.
    Return None if there is no n-of-a-kind in the hand."""

    # generator
    # try:
    #     first_rank = next(x for x in ranks if ranks.count(x) == n)
    #     return first_rank
    # except StopIteration: # no rank of kind n
    #     return None

    return next((x for x in ranks if ranks.count(x) == n), None)


def two_pair(ranks):
    """If there are two pair, return the two ranks as a
    tuple: (highest, lowest); otherwise return None."""
    kind_generator = (x for x in set(ranks) if ranks.count(x) == 2)

    first_pair = next(kind_generator, None)
    second_pair = next(kind_generator, None)

    pairs = (first_pair, second_pair)

    if None in pairs:
        return None

    else:
        return pairs


def best_hand(hand):
    """From a 7-card hand, return the best 5 card hand."""

    five_card_hands = itertools.combinations(hand, 5)

    # best_rank = (-1, None)
    # best_h = None
    # for h in five_card_hands:
    #     rank = hand_rank(h)
    #     if rank > best_rank:
    #         best_rank = rank
    #         best_h = h

    return max(five_card_hands, key=hand_rank)


def best_wild_hand(hand):
    "Try all values for jokers in all 5-card selections."

    five_card_hands = itertools.combinations(hand, 5)

    hands = [find_best_joker(h) for h in five_card_hands]

    return max(hands, key=hand_rank)


def find_best_joker(hand):

    # red_deck = [r+s for r in '23456789TJQKA' for s in 'HD']
    # black_deck = [r+s for r in '23456789TJQKA' for s in 'SC']
    # best_rank = (-1, None)
    # best_h = hand

    # if '?B' in hand or '?R' in hand:
    #     for black_card in black_deck:
    #         for red_card in red_deck:
    #             new_hand = list(map(lambda x: black_card if x == '?B' else \
    #                                             red_card if x == '?R' else x, hand))
    #             rank = hand_rank(new_hand)
    #             if rank > best_rank:
    #                 best_rank = rank
    #                 best_h = new_hand[:]

    # return best_h

    if not ('?B' in hand or '?R' in hand):
        return hand

    deck_map = {
        '?B': [r + s for r in '23456789TJQKA' for s in 'SC'],
        '?R': [r + s for r in '23456789TJQKA' for s in 'HD']
    }

    options = [
        deck_map[card] if card in deck_map else [card]
        for card in hand
    ]

    hands = itertools.product(*options)

    return list(max(hands, key=hand_rank))
