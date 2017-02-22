import deeprl_hw1.queue_envs as queue_envs
import gym


def main():
    env = gym.make('Queue-1-v0')

    next_state = env.reset()
    env.render()

    while True:
        model = env.query_model(next_state, 3)
        print model
        nb = input('Enter input: ')
        next_state, reward,_,_ = env.step(nb)
        env.render()
        


if __name__ == '__main__':
    main()