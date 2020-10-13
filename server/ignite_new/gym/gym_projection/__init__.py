import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Projection-v0',
    entry_point='gym_projection.envs:ProjectionEnv',
    reward_threshold=0.8,
    nondeterministic = True,
)
