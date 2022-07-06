from metaworld_door_open import make_dooropen_env
from metaworld_door_open.policy import SawyerDoorOpenV2Policy
import imageio as iio


env = make_dooropen_env(max_episode_length=200, seed=314)
policy = SawyerDoorOpenV2Policy()

images = []

obs = env.reset()
for i in range(200):
    a = policy.get_action(obs)
    obs, r, done, info = env.step(a)
    img = env.render(mode='rgb_array')
    images.append(img)
    
iio.mimwrite('sample_episode.mp4', images, format='mp4', fps=20)
