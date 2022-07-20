import pickle
import matplotlib.pyplot as plt
import scipy.integrate as it

path_to_reward = '/home/bio/bioinspired_ws/src/tutorial_5/policy/saved_reward.obj'
with open (path_to_reward, 'rb') as f:
    reward = pickle.load(f)


cum_reward = it.cumtrapz(reward,initial=0)
fig, ax = plt.subplots(1,1)
ax.plot(cum_reward)
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative reward')
ax.set_title('Cumulative reward per episode')

plt.savefig('Cumulative_reward_per_episode.png')
plt.show()