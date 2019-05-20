import sys
from agents import QAgent
from players import Player
from environment import Game
from copy import deepcopy


def test(test_env, train, test, test_step):
    """
    Tests an environment against a test agent
    """

    # when learn is True, we need to switch
    restart_learn = train.learning == True

    train.learning = False
    test_env.player1 = train
    test_env.player2 = test
    test_winners = []
    games = []
    states = []
    scores = []

    for test_game in range(test_step):
        game, winner, game_length, state_log, final_score = test_env.play(log=True)
        test_winners.append(winner)
        games.append(game)
        states.append(state_log)
        scores.append(final_score)
        p1 = test_env.player1
        p2 = test_env.player2
        test_env.player1 = p2
        test_env.player2 = p1

    win_percentage = float(test_winners.count(train.name)) / test_step
    draw_percentage = float(test_winners.count('None')) / test_step
    loss_percentage = float(test_winners.count(test.name)) / test_step

    print("Current win % over agent {}: {:.2f}%".format(test.name, win_percentage * 100))
    print("Current loss % over agent {}: {:.2f}%".format(test.name, loss_percentage * 100))
    print("Current draw % over agent {}: {:.2f}%".format(test.name, draw_percentage * 100))

    if restart_learn:
        train.learning = True

    return win_percentage, draw_percentage, loss_percentage


def train(dqn, environment, learning_agent, opponent_agent, iterations, test_agent=None, test_steps=None):
        """
        Training an environment
        """
        # Create a test environment
        if test_agent is not None:
            test_env = deepcopy(environment)

        # Set the players to the environment
        environment.player1 = learning_agent
        environment.player2 = opponent_agent

        if dqn:
            learning_agent.initialize_network()
            opponent_agent.initialize_network()


        game_start = 1
        print ("\nStarting Game at {}".format(game_start))

        # Begin training games
        for game_number in range(game_start, iterations + 1):

            # Switch who goes first every other round
            environment.player1 = learning_agent
            environment.player2 = opponent_agent
            if game_number % 2 == 0:
                p1 = environment.player1
                p2 = environment.player2
                environment.player1 = p2
                environment.player2 = p1

            environment.play()

            # Log game for every test step
            if game_number in test_steps and test_agent:
                print("Game {} Test Results".format(game_number))
                win_percentage, draw_percentage, loss_percentage = test(test_env, learning_agent, test_agent, game_number)
                print('{},{},{},{},{}\n'.format(game_number, test_agent, win_percentage, draw_percentage, loss_percentage))

        print ("Finished!")


if __name__ == '__main__':
    size = 2    # 2 or 3

    size = int(input('Enter Cell Size (Eg. 2 or 3) : '))
    use_dqn = int(input('Choose 1 or 2 --> (1) QTable (2) DQN : '))

    dqn = False
    if use_dqn == 2:
        dqn = True

    print('Game size {} x {}'.format(size, size))
    learning_agent = QAgent('learning',alpha=1e-6,gamma=0.6,dqn=dqn)
    opponent_agent = QAgent('opponent',learning=False,dqn=dqn)

    test_agent = Player(name='random')

    env = Game(size)

    iterations = 10000
    test_steps = [100, 1000, 10000]


    train(dqn,
          env,
          learning_agent,
          opponent_agent,
          iterations,
          test_agent,
          test_steps)
