
class Player:
    name = None
    stack = 0
    is_sitting_in = False
    hole_cards = None

    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hole_cards = []

    def sit_in(self):
        self.is_sitting_in = True

    def sit_out(self):
        self.is_sitting_in = False

    def has_cards(self):
        return self.hole_cards != []

    def add_hole_card(self, card):
        self.hole_cards.append(card)

    def fold(self):
        self.hole_cards = []

    def print_state(self):
        print 'Name: %s' % self.name
        print 'Stack: %s' % self.stack
        print 'Sitting: %s' % ('IN' if self.is_sitting_in else 'OUT')
        print 'Hole Cards: %s' % str(self.hole_cards)


from random import shuffle

class Deck:

    cards = ['As', 'Ks', 'Qs', 'Js', 'Ts', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s',
             'Ah', 'Kh', 'Qh', 'Jh', 'Th', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h',
             'Ad', 'Kd', 'Qd', 'Jd', 'Td', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d',
             'Ac', 'Kc', 'Qc', 'Jc', 'Tc', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c']

    def __init__(self, shuffled=False):
        if shuffled:
            shuffle(self.cards)

    def deal_one(self):
        return self.cards.pop(0)

    def deal_three(self):
        first = self.cards.pop(0)
        second = self.cards.pop(0)
        third = self.cards.pop(0)
        return [first, second, third]


class Stake:
    
    SB = None
    BB = None
    ante = None

    def __init__(self, stake_str):

        stake_split = stake_str.split('/')
        self.SB = float(stake_split[0])
        self.BB = float(stake_split[1])

        if len(stake_split) == 3:
            self.ante = float(stake_split[2])

