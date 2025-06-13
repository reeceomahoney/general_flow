import gymnasium as gym

from .lift_env_cfg import FrankaLiftEnvCfg
from general_flow.config import rsl_rl_ppo_cfg


gym.register(
    id="Isaac-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaLiftEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LiftCubePPORunnerCfg,
    },
)
