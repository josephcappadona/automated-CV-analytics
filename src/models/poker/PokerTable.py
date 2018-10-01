from Utils import Player, Deck
from random import choice


class PokerTable:
    
    seats = None

    num_sitting_in = 0
    num_with_cards = 0

    deck = None
    button = None
    SB_seat = None
    BB_seat = None

    board = []
    pot = 0


    def __init__(self, num_seats):
        self.num_seats = num_seats
        self.seats = [None for _ in range(num_seats)]
        self.open_seats = set(i for i in range(num_seats))
        self.taken_seats = set()

    
    def add_player(self, name, stack, seat):
        if seat in self.open_seats:
            player = Player(name, stack)

            self.open_seats.remove(seat)
            self.taken_seats.add(seat)
            self.seats[seat] = player

        else:
            # throw SeatNotOpenError
            pass

    def remove_player(self, seat):
        player = self.seats[seat]
        if player is not None:

            if player.is_sitting_in:
                self.num_sitting_in -= 1

            self.taken_seats.remove(seat)
            self.seats[seat] = None
            self.open_seats.add(seat)

        return player

    def sit_player_in(self, seat):
        player = self.seats[seat]
        if not player.is_sitting_in:
            player.sit_in()
            self.num_sitting_in += 1

    def sit_player_out(self, seat):
        player = self.seats[seat]
        if player.is_sitting_in:
            player.sit_out()
            self.num_sitting_in -= 1

    def next_player_after(self, seat):
        seat = (seat + 1) % len(self.seats)
        while self.seats[seat] == None:
            seat = (seat + 1) % len(self.seats)
        return seat

    def next_player_sitting_in_after(self, seat):
        seat = self.next_player_after(seat)
        while not self.seats[seat].is_sitting_in:
            seat = self.next_player_after(seat)
        return seat

    def next_player_with_cards_after(self, seat):
        seat = self.next_player_after(seat)
        player = self.seats[seat]
        while not player.has_cards():
            seat = self.next_player_after(seat)
            player = self.seats[seat]
        return seat


    def new_deck(self):
        self.deck = Deck(shuffled=True)

    def add_hole_card(self, seat, card):
        self.seats[seat].add_hole_card(card)


    def place_button(self):
        seats_sitting_in = filter(lambda seat: self.seats[seat] is not None and self.seats[seat].is_sitting_in, range(self.num_seats))
        self.button = choice(seats_sitting_in)
        self.SB_seat = self.next_player_sitting_in_after(self.button)
        self.BB_seat = self.next_player_sitting_in_after(self.SB_seat)

    def move_button(self):
        self.button = self.SB_seat
        self.SB_seat = self.BB_seat
        self.BB_seat = self.next_player_sitting_in_after(self.BB_seat)


    def print_state(self):
        print 'Board: %s' % self.board
        print 'Pot: %s' % self.pot
        print 'Button: %s' % self.button
        for seat, player in enumerate(self.seats):
            print '\nSeat %s:' % seat
            if player is not None:
                player.print_state()
            else:
                print 'Empty Seat'
        print '\n'

