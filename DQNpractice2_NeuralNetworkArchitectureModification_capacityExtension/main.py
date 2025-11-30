# Cartpole 기반의 원래 메인 함수 변경점
# 1. (NUM_STATES=6 기반) states 6개로 변경
# 2. env에서 import 필요
# 3. Cartpole 기반 불필요한 내용 삭제(정의되지 않은 reward, 다른 states)
from matplotlib import pyplot as plt
from dqn import DQN, MEMORY_CAPACITY
from func import trans


def main():
    import env
    myenv=env.Maze(1)

    dqn = DQN()
    episodes = 4000
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    
    for i in range(episodes):
        state = myenv.reset()
        ep_reward = 0
        
        while True:
            #myenv.render()
            action, _ = dqn.choose_action(state)
            real_action = trans(action)
            next_state, reward, time_cost = myenv.step(real_action, 0)
            done = True

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
            
        reward_list.append(ep_reward)
        ax.set_xlim(0,4000)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='reward')
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()