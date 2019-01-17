from PokerTable import PokerTable
from Utils import Stake

class PokerGame:

    table = None

    def __init__(self, stake, num_seats):
        self.table = PokerTable(num_seats)
        self.stake = Stake(stake)
        self.num_seats = num_seats


    def game_loop(self):

        if self.table.button == None:
            self.table.place_button()
        else:
            self.table.move_button()

        self.table.new_deck()

        # Pre-Flop
        self.deal_hole_cards()
        self.table.print_state()
        winner, pot = self.betting_round(preflop=True)
        if winner is not None:
            return (winner, pot)
        
        # Flop
        self.deal_flop()
        self.table.print_state()
        winner, pot = self.betting_round()
        if winner is not None:
            return (winner, pot)

        # Turn
        self.deal_turn()
        self.table.print_state()
        winner, pot = self.betting_round()
        if winner is not None:
            return (winner, pot)

        # River
        self.deal_river()
        self.table.print_state()
        winner, pot = self.betting_round()
        if winner is not None:
            return (winner, pot)

        self.table.print_state()
        # determine winner
        winner = int(raw_input('Who won? '))
        sel.award_pot(self, winner)


    def deal_hole_cards(self):

        # find the first player after the button
        seat = self.table.next_player_sitting_in_after(self.table.button)
        for _ in range(self.table.num_sitting_in * 2):
            card = self.table.deck.deal_one()
            self.table.add_hole_card(seat, card)
            seat = self.table.next_player_sitting_in_after(seat)
        self.table.num_with_cards = self.table.num_sitting_in

    def deal_flop(self):
        self.table.deck.deal_one() # burn
        self.table.board.extend(self.table.deck.deal_three())

    def deal_turn(self):
        self.table.deck.deal_one() # burn
        self.table.board.append(self.table.deck.deal_one())

    def deal_river(self):
        self.table.deck.deal_one() # burn
        self.table.board.append(self.table.deck.deal_one())

    def betting_round(self, preflop=False):

        action = []
        current_bets = [0 for _ in range(self.num_seats)]
        prev_bet = 0
        current_bet = 0

        seat = self.table.button
        if preflop:
            if self.stake.ante is not None:
                players_who_ante = self.post_antes()
                action.extend([(player, 'ANTE %d' % self.stake.ante) for player in players_who_ante])
            
            seat = self.table.next_player_with_cards_after(seat)
            self.table.SB_seat = seat
            self.post_small_blind(seat)
            action.append((seat, 'SB %d' % self.stake.SB))
            current_bets[seat] = self.stake.SB

            seat = self.table.next_player_with_cards_after(seat)
            self.table.BB_seat = seat
            self.post_big_blind(seat)
            action.append((seat, 'BB %d' % self.stake.BB))
            current_bets[seat] = self.stake.BB
            current_bet = self.stake.BB

        players_left_to_act = self.table.num_with_cards
        seat = self.table.next_player_with_cards_after(seat) # first to act
        while players_left_to_act > 0:

            seat_action = raw_input('%s: ' % seat)
            action.append((seat, seat_action))

            if seat_action[0] in set(['B', 'R', 'A']):
                if seat_action[0] == 'A':
                    bet_size = self.table.seats[seat].stack
                else:
                    bet_size = float(seat_action.split(' ')[-1])
                if not self.is_valid_bet(seat, bet_size, current_bet, prev_bet, current_bets):
                    continue
                self.player_bets(seat, bet_size, current_bets)
                prev_bet = current_bet
                current_bets[seat] = bet_size
                current_bet = bet_size
                players_left_to_act = self.table.num_with_cards - 1

            elif seat_action[0] == 'C':
                if current_bet - current_bets[seat] == 0:
                    continue
                self.player_calls(seat, current_bet, current_bets)
                current_bets[seat] = current_bet
                players_left_to_act -= 1

            elif seat_action[0] == 'X':
                if current_bet - current_bets[seat] != 0:
                    continue
                self.player_checks(seat)
                players_left_to_act -= 1

            elif seat_action[0] == 'F':
                self.player_folds(seat)
                self.table.num_with_cards -= 1
                players_left_to_act -= 1

            seat = self.table.next_player_with_cards_after(seat) # next to act

            if self.table.num_with_cards == 1:
                winner = self.table.next_player_with_cards_after(seat)
                pot = self.table.pot
                self.award_pot(winner)
                return (winner, pot)

        return (None, self.table.pot)


    def post_antes(self, ante): 
        players_who_ante = []

        seat = self.table.next_player_with_cards_after(self.table.button)
        while len(players_who_ante) != len(self.table.players_with_cards):
            players_who_ante.append(seat)
            self.player_posts(seat, ante)
            seat = self.table.next_player_with_cards_after(self.table.button)

        return players_who_ante

    def post_small_blind(self, seat):
        self.player_posts(seat, self.stake.SB)

    def post_big_blind(self, seat):
        self.player_posts(seat, self.stake.BB)

    def player_posts(self, seat, num_chips):
        self.table.seats[seat].stack -= num_chips
        self.table.pot += num_chips

    def player_bets(self, seat, bet_size, current_bets):
        bet_on_top = bet_size - current_bets[seat]
        self.table.seats[seat].stack -= bet_on_top
        self.table.pot += bet_on_top

    def is_valid_bet(self, seat, bet_size, current_bet, prev_bet, current_bets):
        is_valid = bet_size > current_bet \
                       and bet_size - current_bet >= current_bet - prev_bet \
                       and bet_size - current_bets[seat] <= self.table.seats[seat].stack \
                       and bet_size >= self.stake.BB
        return is_valid

    def player_calls(self, seat, current_bet, current_bets):
        bet_on_top = current_bet - current_bets[seat]
        self.table.seats[seat].stack -= bet_on_top
        self.table.pot += bet_on_top

    def player_folds(self, seat):
        self.table.seats[seat].fold()

    def player_checks(self, seat):
        pass


    def award_pot(self, seat):
        self.table.seats[seat].stack += self.table.pot
        self.table.pot = 0
