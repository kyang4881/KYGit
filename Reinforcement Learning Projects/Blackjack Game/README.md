<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Reinforcement%20Learning%20Projects/Blackjack%20Game/docs/images/blackjack.jpg" width="1200" />
</p>

---

In the game of Blackjack, a modified version of the classic casino game is played where there are no Kings in the deck, and the cards range from Ace to Queen. The objective is to train an agent using reinforcement learning techniques to play Blackjack and optimize winnings or minimize losses when facing the House. The environment is defined with number cards having their face value, pictures (Jack and Queen) counting as 10, and Aces being flexible with values of 1 or 11. A "blackjack" occurs when a player reaches 21 on the first two cards, the player wins unless the dealer also has a "blackjack." Wins are paid 1 to 1, while player "blackjacks" are paid 3 to 2 odds. The player can either "hit" (take a card) or "stand" (end turn) with the goal of not exceeding 21 points to avoid a "bust." The dealer, representing the house, follows specific rules for drawing cards based on their hand's value.

---

## Overview

* Deck: Ace, 2-10, Jack and Queen
* Objective: Maximize winnings or minimize losses

Env:

* Ace: {1, 11}, Jack or Queen = 10, else = face value
* Win if total = 21 on the first 2 cards = blackjack
* Tie if dealer also has blackjack
* Win payout = +1, blackjack payout = +1.5, tie = 0
* Player is dealt two cards, action = {"hit", "stand"}
* Card total > 21 = busts (loses)
* Dealer has 1 face up card and draw iff hard 16, soft 17 or lower

Agents:

* Q Learning
* Monte Carlo

---

## Environment

The class BlackjackEnvironment simulates a game of Blackjack. The class initializes the game environment with a deck of cards, assigns values to each card in the deck, and offers methods for simulating the game. The deal_card method simulates drawing a card from the deck, and the card_mapper method maps card values to their corresponding numeric values. The initial_state method initializes the game state, generating the two player cards and one face up dealer card. The player can choose to "hit" (draw a card) or "stand" (end their turn) using the player_action method. The hand_total methods calculate the total value of a hand, considering possible values for Aces. The dealer_action method determines the dealer's action based on their hand, adhering to typical Blackjack rules. The accepted_value method helps identify acceptable values for the player and dealer, ensuring values do not exceed 21. The step method simulates a single step in the game, updating player and dealer hands, calculating rewards, and determining if the game is over. The play method simulates an entire game and returns the final reward.

The following assumptions were implemented in the environment and state-transition.
* Deck Composition: The deck is modified to exclude Kings, leaving only Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, and Queen.
    * The provided code defines a Python class called BlackjackEnvironment, which simulates a simplified game of Blackjack. The class initializes the game environment with a deck of cards as defined in the assumption.

* Card Values: Number cards are considered at face value, Jack and Queen cards are both valued at 10, and Aces can be counted as either 1 or 11.
    * A card_mapper method maps card values to their corresponding numeric values

* Blackjack Definition: A player achieving a total of 21 on the first two cards is considered to have a "blackjack." In this case, the player wins unless the dealer also has a "blackjack," leading to a "tie."

    * In the transition method, the player goes first and it is checked whether the player has 21, if the player has a total of 21 and the dealer doesn't, it's a blackjack and the turn ends. If the dealer also has 21, it's a tie game, otherwise the game continues until the player either stand or bust.

* Payouts: Payouts for regular wins are simplified at 1 to 1, and player "blackjacks" are paid out at 3 to 2 odds.
    * Rewards are assigned in the transition method with +1.5 for blackjack, +1 for a normal win, 0 for a tie and -1 for a loss.

* Player Actions: The agent is given two possible actions: "hit" (take a card) or "stand" (end the turn). This simplifies the agent's decision-making process.
    * The player's action is determined by the Agent class and consists of only the two listed actions.

* Dealer Rules: The dealer (representing the house) has a single faced-up card, and they must draw an additional card if their hand is currently "hard 16" or "soft 17" (which includes an Ace) or lower.
    * This is implemented by the dealer_action method

```python
class BlackjackEnvironment:
    def __init__(self, verbose=False):
        """
        Initialize the Blackjack environment.
        Args:
            verbose (bool): If True, print additional information during the game.
        """
        # Define the deck
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Ace, 2-10, Jack, Queen
        self.card_value = {}
        self.verbose = verbose

        # Assign values to cards in the deck
        for c in self.deck:
            if c == 1:
                self.card_value[c] = [c, 11]  # Ace can be 1 or 11
            elif c in np.arange(2, 11):
                self.card_value[c] = c # Cards 2-10 have their face values
            else:
                self.card_value[c] = 10 # Face cards (Jack, Queen) have a value of 10

    def deal_card(self):
        """
        Simulate dealing a card from the deck.

        Returns:
            int: A randomly chosen card from the deck.
        """
        # Simulate dealing a card from the deck.
        return np.random.choice(self.deck)

    def card_mapper(self, card_hand):
        """
        Map cards in a hand to their corresponding values.

        Args:
            card_hand (list of int): A list of cards in the hand.

        Returns:
            list of int: A list of card values.
        """
        # Map cards in a hand to their corresponding values
        hand = []
        for card in card_hand:
            hand.append(self.card_value[card])
        return hand

    def initial_state(self):
        """
        Initialize the initial state of the game with player and dealer hands.

        Returns:
            tuple of lists: Player and dealer hands as lists.
        """
        player_hand = [self.deal_card(), self.deal_card()]
        #dealer_hand = [self.deal_card(), self.deal_card()]
        dealer_hand = [self.deal_card()]
        return player_hand, dealer_hand

    def player_action(self, player_hand):
        # The player can choose to hit or stand
        return np.random.choice(['hit', 'stand'])

    def hand_total_original(self, hand):
        """
        Calculate the total value of a hand, considering possible values for Aces.

        Args:
            hand (list of int): A list of card values in the hand.

        Returns:
            Total values with and without considering Aces, and a flag for the presence of Ace.
        """
        mapped_hand = self.card_mapper(hand)
        combo_1 = 0
        combo_2 = 0
        has_ace = False
        for card in mapped_hand:
            if card == [1, 11]:
                combo_1 += 1
                combo_2 += 11
                has_ace = True
            else:
                combo_1 += card
                combo_2 += card
        return combo_1, combo_2, has_ace

    def hand_total(self, hand):
        """
        Calculate the total value of a hand, considering possible values for Aces.

        Args:
            hand (list of int): A list of card values in the hand.

        Returns:
            Tuple containing the soft (with Ace as 11) and hard (with Ace as 1) hand totals and a flag for the presence of Ace.
        """
        mapped_hand = self.card_mapper(hand)
        soft_total = 0
        hard_total = 0
        has_ace = False

        for card in mapped_hand:
            if card == [1, 11]:
                if soft_total + 11 <= 21:
                    soft_total += 11  # Ace as 11 in soft total
                    hard_total += 1  # Ace as 1 in hard total
                else:
                    soft_total += 1  # Ace as 1 in soft total
                    hard_total += 1  # Ace as 1 in hard total
                has_ace = True
            else:
                soft_total += card
                hard_total += card

        return hard_total, soft_total, has_ace

    def dealer_action(self, dealer_hand):
        """
        Determine the dealer's action (hit or stand).

        Args:
            dealer_hand (list of int): Dealer's current hand.

        Returns:
            str: The selected action, either 'hit' or 'stand'.
        """
        combo_1, combo_2, has_ace = self.hand_total(dealer_hand)

        hard_16 = False
        soft_17 = False
        lower = False

        if combo_1 == 16 and has_ace == False:  # hard 16
            hard_16 = True;
        elif combo_2 == 17 and has_ace:  # soft 17
            soft_17 = True ;
        elif min(combo_1, combo_2) <= 16 and hard_16 == False and soft_17 == False: # lower value
            lower = True;
        else:
            pass

        if hard_16 or soft_17 or lower:
            return "hit"
        else:
            return "stand"

    def accepted_value_original(self, player_values, dealer_values):
        """
        Determine the accepted values for both player and dealer.

        Args:
            player_values (list of int): Possible total values for the player.
            dealer_values (list of int): Possible total values for the dealer.

        Returns:
            list of int and list of int: Accepted values for the player and dealer, considering values not exceeding 21.
        """
        # Return list of values in the player's and dealer's hands that are less than or equal to 21
        return list(set(v for v in player_values if v <= 21)), list(set(d for d in dealer_values if d <= 21))

    def accepted_value(self, cards):
        """
        Determine the accepted values for both player and dealer.

        Args:
            cards (list of int): Possible total values for the cards.

        Returns:
            list of int and list of int: Accepted values for the player and dealer, considering values not exceeding 21.
        """
        # Return list of values in the player's and dealer's hands that are less than or equal to 21
        return list(set(v for v in cards if v <= 21))

    def step(self, player_hand, dealer_hand, player_action):
        """
        Simulate a single step in the game.

        Args:
            player_hand (list of int): Player's current hand.
            dealer_hand (list of int): Dealer's current hand.
            player_action (str): The player's chosen action, either 'hit' or 'stand'.

        Returns:
            tuple: New player hand, dealer hand, reward, and a flag indicating if the game is done.
        """
        player_combo_1, player_combo_2, _ = self.hand_total(player_hand)
        dealer_combo_1, dealer_combo_2, _ = self.hand_total(dealer_hand)
        dealer_max = max(dealer_combo_1, dealer_combo_2)
        dealer_min = min(dealer_combo_1, dealer_combo_2)
        player_max = max(player_combo_1, player_combo_2)
        player_min = min(player_combo_1, player_combo_2)

        # Player has a blackjack, and the dealer does not
        if (player_max == 21 or player_min == 21) and len(player_hand) == 2 and (dealer_max != 21 and dealer_min != 21):
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: stand")
                print("Player wins (BlackJack).\nRewards: +1.5\n")
            return player_hand, dealer_hand, 1.5, True  # Player wins; black jack

        # Both player and dealer have blackjack
        elif (player_max == 21 or player_min == 21) and len(player_hand) == 2 and (dealer_max == 21 or dealer_min == 21):
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: stand")
                print("Tie (Both BlackJack).\nRewards: 0\n")
            return player_hand, dealer_hand, 0, True # Tie

        else:
            # Player doesn't have a blackjack
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: {player_action}")
            # Player chooses to hit
            if player_action == 'hit':
                player_hand.append(self.deal_card())
                player_combo_1, player_combo_2, _ = self.hand_total(player_hand)
                player_max = max(player_combo_1, player_combo_2)
                player_min = min(player_combo_1, player_combo_2)

                # Player busts, loses
                if player_min > 21:
                    if self.verbose and count / 2**power == 1:
                        print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                        print("Player loses (Busts).\nRewards: -1\n")
                    return player_hand, dealer_hand, -1, True

                else:
                    return player_hand, dealer_hand, 0, False
            else:
                # Player stands, and it's the dealer's turn
                dealer_action = self.dealer_action(dealer_hand)
                while True:
                    dealer_action = self.dealer_action(dealer_hand)
                    # Dealer chooses to hit
                    if dealer_action == 'hit':
                        if self.verbose and count / 2**power == 1:
                            print(f"Dealer:\n---Hand: {dealer_hand}\n---Max Value: {dealer_max}\n---Min Value: {dealer_min}")
                            print(f"Dealer Action: {dealer_action}")
                        dealer_hand.append(self.deal_card())
                        dealer_combo_1, dealer_combo_2, _ = self.hand_total(dealer_hand)
                        dealer_max = max(dealer_combo_1, dealer_combo_2)
                        dealer_min = min(dealer_combo_1, dealer_combo_2)
                    # Dealer stands
                    else:
                        #player_accepted_val, dealer_accepted_val = self.accepted_value(player_values=[player_combo_1, player_combo_2], dealer_values=[dealer_combo_1, dealer_combo_2])
                        player_accepted_val = self.accepted_value([player_combo_1, player_combo_2])
                        dealer_accepted_val = self.accepted_value([dealer_combo_1, dealer_combo_2])

                        if self.verbose and count / 2**power == 1:
                            print(f"Dealer:\n---Hand: {dealer_hand}\n---Max Value: {dealer_max}\n---Min Value: {dealer_min}")
                            print(f"Dealer Action: {dealer_action}")
                        # Dealer busts, player wins
                        if dealer_min > 21:
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Dealer Busts).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                        # Player wins, has a higher total
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)>0 and max(player_accepted_val) > max(dealer_accepted_val):
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Better Hand).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                         # Player wins
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)==0:
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Better Hand).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                        # Tie game
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)>0 and max(player_accepted_val) == max(dealer_accepted_val):
                            if self.verbose and count / 2**power == 1:
                                print("Tie.\nRewards: 0\n")
                            return player_hand, dealer_hand, 0, True
                        # Player loses
                        else:
                            if self.verbose and count / 2**power == 1:
                                print("Player loses (Worse Hand).\nRewards: -1\n")
                            return player_hand, dealer_hand, -1, True

    def play(self):
        """
        Simulate a complete game and return the final reward.

        Returns:
            float: The reward for the game.
        """
         # Simulate a complete game and return the final reward
        player_hand, dealer_hand = self.initial_state()
        done = False
        while not done:
            player_action = self.player_action(player_hand)
            player_hand, dealer_hand, reward, done = self.step(player_hand, dealer_hand, player_action)
            #print(player_hand, dealer_hand, reward, done)
        return reward

```

### Method 1 - Q Learning Agent

This agent is designed to learn a policy that helps maximize the expected cumulative reward in a simplified Blackjack environment based on Q Learning.

The choose_action method is responsible for selecting the agent's action (either "hit" or "stand") based on the Q-values associated with the current player hand. It implements an epsilon-greedy policy, where with probability epsilon, the agent explores by choosing a random action, and with probability 1 - epsilon, it exploits by selecting the action with the highest Q-value for the current state.

The learn method updates the Q-values based on the observed reward and the next state. It uses the Q-learning update formula to compute the updated Q-value, where it considers the current Q-value, the observed reward, and the maximum Q-value of the next state. The Q dictionary is updated accordingly.

The train method is responsible for training the Q-learning agent by simulating Blackjack games and updating Q-values based on the game outcomes. It iterates through a specified number of training episodes and, for each episode, it initializes the player and dealer hands, simulates the game, and updates Q-values using the choose_action and learn methods.

```python
class QLearningAgent:
    """
    Initialize the Q-learning agent.

    Args:
        env (BlackjackEnvironment): The environment in which the agent operates.
        alpha (float): Learning rate for Q-learning.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Epsilon-greedy exploration parameter.
        num_episodes (int): The number of episodes for training.
        Q (dict): A dictionary containing q values for the states
        verbose (bool): If True, print additional information during training and testing.
    """
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=100000, verbose=False):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy exploration parameter
        self.num_episodes = num_episodes
        self.Q = {}
        self.verbose = verbose

    def choose_action(self, player_hand):
        """
        Choose an action (hit or stand) based on the Q-values for the given player hand.

        Args:
            player_hand (list of int): Player's current hand.

        Returns:
            str: The selected action, either 'hit' or 'stand'.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.player_action(player_hand)  # Explore (random action)
        else:
            # Exploit (choose action with highest Q-value)
            state = tuple(player_hand)
            if state not in self.Q:
                self.Q[state] = {'hit': 0, 'stand': 0}

            hit_value = self.Q[state]['hit']
            stand_value = self.Q[state]['stand']

            if self.verbose and count / 2**power == 1:
                print(f"Q_values:\n---Hit: {float(hit_value)}\n---Stand: {float(stand_value)}")
            if hit_value > stand_value:
                return 'hit'
            elif hit_value == stand_value:
                # Randomly choose between 'hit' and 'stand' when they are the same
                return random.choice(['hit', 'stand'])
            else:
                return 'stand'

    def learn(self, player_hand, player_action, reward, player_hand_next):
        """
        Update the Q-value for a given state-action pair based on the observed reward and next state.

        Args:
            player_hand (list of int): Player's current hand.
            player_action (str): The action taken by the player, either 'hit' or 'stand'.
            reward (float): The reward obtained for taking the specified action.
            player_hand_next (list of int): Player's hand in the next state.
        """
        # Update Q-value using Q-learning formula
        state = tuple(player_hand)
        if state not in self.Q:
            self.Q[state] = {'hit': 0, 'stand': 0}
        # Current Q value
        Q_current = self.Q[state][player_action]
        state_next = tuple(player_hand_next)
        # Optimal future Q value
        Q_future = max(self.Q[state_next]['hit'], self.Q[state_next]['stand'])
        updated_value = (1 - self.alpha) * Q_current + self.alpha * (reward + self.gamma * Q_future)
        # Update Q table
        self.Q[state][player_action] = updated_value

    def train(self):
        """
        Train the Q-learning agent by simulating blackjack games and updating Q-values based on the results.
        """
        global count
        global power
        count = 0
        power = 0
        for episode in range(self.num_episodes):
            player_hand, dealer_hand = self.env.initial_state()
            done = False
            if self.verbose and count / 2**power == 1:
                print(f"Episode: {episode}")
            while not done:
                player_action = self.choose_action(player_hand)
                player_hand_next, dealer_hand_next, reward, done = self.env.step(player_hand, dealer_hand, player_action)
                self.learn(player_hand, player_action, reward, player_hand_next)
                player_hand, dealer_hand = player_hand_next, dealer_hand_next
                #if self.verbose and count / 2**power == 1 and done:
                    #power += 1
            #count += 1
    
```

This method is used to test the performance of the Q-learning agent.

```python
def test(env, agent, verbose=False, num_episodes=20000):
    """
    Test the Q-learning agent by playing blackjack games and measuring the average reward over multiple episodes.

    Args:
        env (obj): The environment
        agent (obj): The input agent model
        num_episodes (int): The number of episodes for testing.

    Returns:
        list of float and float: List of rewards for each episode and the average reward.
    """
    global count
    global power
    count = 0
    power = 0
    rewards = []
    win_rates = []
    draw_rates = []
    tracker = {"player_hand": [], "dealer_showing" : [], "player_has_ace": [], "dealer_has_ace": [], "player_action": []}

    for episode in range(num_episodes):
        player_hand, dealer_hand = env.initial_state()
        #if verbose: print("player_hand, dealer_hand: ", player_hand, dealer_hand)

        done = False
        episode_reward = 0

        if verbose and count / 2**power == 1:
            print(f"Episode: {episode}")
            #print(tracker)
            if len(win_rates) > 0 and len(draw_rates) > 0:
                print(f"Win rate: {win_rate}, Draw rate: {draw_rate}")
                # Plot winning rates
                plt.close()
                plt.plot(win_rates)
                plt.show()

        while not done:
            player_action = agent.choose_action(player_hand)

            # Compile a dictionary containing player hand and action, dealer showing, and whether or not player/dealer has ace
            player_c1, player_c2, player_has_ace = env.hand_total(player_hand)
            player_acceted_value = 22 if len(env.accepted_value([player_c1, player_c2])) == 0 else max(env.accepted_value([player_c1, player_c2]))
            dealer_showing = env.hand_total([dealer_hand[0]])[0]
            dealer_has_ace = True if env.hand_total([dealer_hand[0]])[0] == 1 or env.hand_total([dealer_hand[0]])[1] == 1 else False
            tracker["player_hand"].append(player_acceted_value)
            tracker["dealer_showing"].append(dealer_showing)
            tracker["player_has_ace"].append(player_has_ace)
            tracker["dealer_has_ace"].append(dealer_has_ace)
            tracker["player_action"].append(player_action)

            player_hand_next, dealer_hand_next, reward, done = env.step(player_hand, dealer_hand, player_action)

            # Player busts; update the tracking dictionary
            player_c1_next, player_c2_next, player_has_ace = env.hand_total(player_hand_next)
            player_acceted_value_next = 22 if len(env.accepted_value([player_c1_next, player_c2_next])) == 0 else max(env.accepted_value([player_c1_next, player_c2_next]))
            if player_acceted_value_next == 22:
                tracker["player_hand"][-1] = player_acceted_value_next

            player_hand, dealer_hand = player_hand_next, dealer_hand_next
            episode_reward += reward
            if verbose and count / 2**power == 1 and done:
                power += 1
        count += 1
        rewards.append(episode_reward)

        if len(rewards) > 0:
            # Calculate win/draw rate
            rewards_dict = dict(Counter(rewards))
            win_rate = (rewards_dict.get(1, 0) + rewards_dict.get(1.5, 0)) / sum(rewards_dict.values())
            draw_rate = rewards_dict.get(0, 0) / sum(rewards_dict.values())
            # Save the rates
            win_rates.append(win_rate)
            draw_rates.append(draw_rate)

    avg_reward = np.mean(rewards)
    print(f"Win Rate: {win_rates[-1]}, Draw Rate: {draw_rates[-1]}, Average reward over {num_episodes} episodes: {avg_reward}")
    plt.close()
    plt.plot(win_rates)
    plt.show()
    return rewards, avg_reward, tracker


def plot_policy(tracker):
    player_hand = np.array(tracker["player_hand"])
    dealer_showing = np.array(tracker["dealer_showing"])
    player_action = np.array(tracker["player_action"])

    # Create a grid for the heatmap
    x_range = np.arange(2, 12)
    y_range = np.arange(22, 10, -1)
    X, Y = np.meshgrid(x_range, y_range)

    # Initialize arrays to store hit and stand counts
    hit_counts = np.zeros((len(y_range), len(x_range)))
    stand_counts = np.zeros((len(y_range), len(x_range)))

    for i in range(len(player_hand)):
        if 11 <= player_hand[i] <= 22 and 1 <= dealer_showing[i] <= 11:
            y_idx = 22 - player_hand[i]
            x_idx = dealer_showing[i] - 1

            if player_action[i] == "hit":
                hit_counts[y_idx, x_idx] += 1
            elif player_action[i] == "stand":
                stand_counts[y_idx, x_idx] += 1

    # Calculate the percentages of "hit" and "stand" actions
    total_counts = hit_counts + stand_counts
    hit_percentages = np.where(total_counts > 0, hit_counts / total_counts, 0)
    stand_percentages = np.where(total_counts > 0, stand_counts / total_counts, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(hit_percentages, cmap=plt.get_cmap('Wistia'), extent=[1.5, 11.5, 10.5, 22.5])

    plt.xticks(x_range - 0.5, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.yticks(y_range - 0.5, y_range)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Hand')
    ax.grid(color='black', linestyle='-', linewidth=1)

    #cbar = plt.colorbar(cax, ticks=[0, 0.5, 1])
    #cbar.ax.set_yticklabels(['0% (stand)', '50%', '100% (hit)'])

    # Add text labels for percentages in each cell
    for i in range(len(y_range)):
        for j in range(len(x_range)):
            if total_counts[i, j] > 0:
                text = f"{int(hit_percentages[i, j] * 100)}% H\n{int(stand_percentages[i, j] * 100)}% S"
                ax.annotate(text, (x_range[j] - 0.0, y_range[i] - 0.0), ha='center', va='center', fontsize=8, color='black')

    plt.title("Hit vs. Stand Percentage Heatmap")
    plt.show()


def get_action_cnt(tracker):
    # Check the counts of the player actions by card total
    tracker_dict = Counter(zip(tracker["player_hand"], tracker["player_action"]))
    sorted_items = sorted(tracker_dict.items(), key=lambda item: (item[0][0], item[0][1]))
    # Convert the sorted list back to a Counter dictionary
    sorted_dict = Counter(dict(sorted_items))
    return sorted_dict
```

Instantiate the environment and train the agent

```python
env = BlackjackEnvironment(verbose=True)
agent1 = QLearningAgent(env, num_episodes=1000000, alpha=0.1, gamma=0.9, epsilon=0.1, verbose=True)
agent1.train()
```

Benchmark the performance of the trained agent

```python
rewards1, avg_reward1, tracker1 = test(env=env, agent=agent1, verbose=True, num_episodes=10000)
```

Visualize the performance

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Reinforcement%20Learning%20Projects/Blackjack%20Game/docs/images/blackjack_agent_performance.png" width="800" />
</p>







A class for fine-tuning a model and saving the predictions.

```python
class finetunedModel:
    """A class for generating results using a finetuned model
    Args:
        repository_id (str): A string id for the repository
        device (str): A string for selecting either cpu or gpu
        model_type (str): A string indicating the type of model
        data_dict (dataset): A dataset containing the train, validation, and test data
        data_selected (str): A string for selecting either the train, validation, or test data
        pred_filepath (str): The folder path to save the predictions
    """
    def __init__(self, repository_id, device, model_type, data_dict, data_selected, pred_filepath):
        self.repository_id = repository_id
        self.device = device
        self.model_type = model_type
        self.data_dict = data_dict
        self.data_selected = data_selected
        self.predicted_labels = []
        self.predicted_labels_int = []
        self.input_data_selected = self.data_dict[self.data_selected]
        self.true_labels = self.input_data_selected["labels"]
        self.true_labels_int = self.input_data_selected["labels_int"]
        self.pred_filepath = pred_filepath

    def load_model(self):
        """A method for loading a fine-tuned model
        Returns:
            A fined-tuned model
        """
        # Load model
        start_time = time.time()
        print(f"Loading model...{self.repository_id}")
        loaded_model = pipeline(self.model_type, model=self.repository_id, device=self.device)
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return loaded_model

    def generate_pred(self, model):
        """Generate the predicted labels
        Args:
            model (obj): A pre-trained model
        Returns:
            A dataframe containing the features, labels, and predicted labels
        """
        start_time = time.time()
        print("Generating predictions...")
        # Prepare input data
        input_data = self.input_data_selected['features']
        # Generate predictions
        for i in range(len(self.input_data_selected['features'])):
            predicted_label = model(input_data[i])[0]['generated_text']
            # Append the binary predicted values for the labels
            if predicted_label == self.input_data_selected['option1'][i]:
                self.predicted_labels_int.append(1)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option2'][i]:
                self.predicted_labels_int.append(2)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option3'][i]:
                self.predicted_labels_int.append(3)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option4'][i]:
                self.predicted_labels_int.append(4)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option5'][i]:
                self.predicted_labels_int.append(5)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            else:
                rand_int = random.randint(1, 5)
                self.predicted_labels_int.append(rand_int)
                if rand_int == 1:
                    self.predicted_labels.append(self.input_data_selected['option1'][i])
                if rand_int == 2:
                    self.predicted_labels.append(self.input_data_selected['option2'][i])
                if rand_int == 3:
                    self.predicted_labels.append(self.input_data_selected['option3'][i])
                if rand_int == 4:
                    self.predicted_labels.append(self.input_data_selected['option4'][i])
                if rand_int == 5:
                    self.predicted_labels.append(self.input_data_selected['option5'][i])

        # Compile results into a dataframe
        res_df = pd.DataFrame({
            'original_sentence': self.input_data_selected['original_sentence'],
            "features": self.input_data_selected['features'],
            "labels": self.input_data_selected['labels'],
            "labels_int": self.input_data_selected['labels_int'],
            "option1": self.input_data_selected['option1'],
            "option2": self.input_data_selected['option2'],
            "option3": self.input_data_selected['option3'],
            "option4": self.input_data_selected['option4'],
            "option5": self.input_data_selected['option5'],
            "predicted_labels": self.predicted_labels,
            "predicted_labels_int": self.predicted_labels_int
        })
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return res_df

    def scoring_metric(self):
        """Generate the accuracy score for the predicted labels
        """
        print("Accuracy: ", accuracy_score(self.true_labels_int, self.predicted_labels_int))

    def save_preds(self, preds):
        """Save predictions to a csv file
        """
        preds['predicted_labels_int'].to_csv(self.pred_filepath + '.csv', index=False, header = False)
        preds.to_excel(self.pred_filepath + '.xlsx', index=False)
        with open(self.pred_filepath + '.txt','w') as f:#, encoding='utf-16-le') as f:
            for p in preds['predicted_labels_int']: f.write(f"{strip(p)}\n")
        print(f"Predictions saved to: {self.pred_filepath}")
```

A function for mapping the data

```python
def map_data(file_name):
    data = pd.read_excel(file_name)
    data['Score'] = [str(s) for s in data['Score']]
    mapping = {'-1.0': 'Dovish', '-0.5':'Mostly Dovish', '0.0': 'Neutral', '0.5': 'Mostly Hawkish', '1.0':'Hawkish'}
    mapping2 = {'-1.0': 1, '-0.5': 2, '0.0': 3, '0.5': 4, '1.0': 5}
    mapped_values = [mapping[value] for value in data['Score']]
    mapped_values2 = [mapping2[value] for value in data['Score']]
    data['answer'] = mapped_values
    data['answer_int'] = mapped_values2
    data_labels = list(mapping.values())
    for i in range(len(data_labels)): data['option' + str(i+1)] = data_labels[i]
    data['sentence'] = [w.replace("_x000D_", "").strip() for w in data['Sentence']]
    return data
```

Prepare the data for training

```python
train_valid_df = map_data(file_name="kenn_fedspeak_20perc_train_small_mod_v5.xlsx")
display(train_valid_df.head())
test_df = map_data(file_name="kenn_fedspeak_20perc_test_small_v5.xlsx")
display(test_df.head())

X_train, X_validation, y_train, y_validation = train_test_split(train_valid_df[['sentence', 'option1', 'option2', 'option3', 'option4', 'option5']], train_valid_df[['answer', 'answer_int']], test_size=0.1, random_state=42, shuffle=True)
X_train = X_train.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_validation = y_validation.reset_index(drop=True)

X_test = test_df[['sentence', 'option1', 'option2', 'option3', 'option4', 'option5']]
y_test = test_df[['answer', 'answer_int']]

print("X_train", np.shape(X_train))
print("X_validation", np.shape(X_validation))
print("X_test", np.shape(X_test))
print("y_train", np.shape(y_train))
print("y_validation", np.shape(y_validation))
print("y_test", np.shape(y_test))

# Compile dataset
data_compiler = compileData(X_train=X_train, X_validation=X_validation, X_test=X_test, y_train=y_train, y_validation=y_validation, y_test=y_test, prompt='What is the most logical completion for the following text?')
nlp_dataset_dict_wtest = data_compiler.compile_dataset()
nlp_dataset_dict_wtest
```
Initialize and start training

```python
# Load a pre-trained checkpoint model
checkpoint = "google/flan-t5-large"
repository_id=checkpoint + "_ky_test_copy_v11"
model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

# Instantiate the preprocessor class
preprocess_data = preprocessor(
    data_dict = nlp_dataset_dict_wtest,
    padding = False,
    truncation = False,
    tokenizer = tokenizer
)
tokenized_dict = preprocess_data.map_inputs()

# Instantiate the trainPipeline class
make_pred = trainPipeline(
    model=model,
    repository_id=repository_id,
    learning_rate= 5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.001,
    save_total_limit=3,
    num_train_epochs=15,
    data_dict=nlp_dataset_dict_wtest,
    padding=True,
    truncation=True,
    tokenizer=tokenizer,
    tokenized_dict=tokenized_dict,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end="True",
    logging_strategy="steps",
    logging_steps=500,
    overwrite_output_dir=False,
    device=0,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42
)
trainer = make_pred.training()

# Train the model (medium)
trainer.train()
trainer.evaluate()
```

Training results printed

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/training_printouts.png" width="1200" />
</p>


Veiw the best model checkpoint

```python
trainer.state.best_model_checkpoint
```

Save the model

```python
# Path to save the best model
best_model_path = repository_id
# Save the tokenizer
tokenizer.save_pretrained(best_model_path)
# Save the model
trainer.save_model(best_model_path)
```


## Sources

1. Hansen, A. and Kazinnik, S., Can ChatGPT Decipher Fedspeak?, March 2023
2. Pan, T., and Lee, H., AI in Finance: Deciphering Fedspeak with Natural Language Processing, March 2021