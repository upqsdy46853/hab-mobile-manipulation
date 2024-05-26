import habitat_extensions.tasks.rearrange
from habitat_baselines.common.baseline_registry import baseline_registry
from mobile_manipulation.config import get_config
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from habitat_extensions.utils.viewer import OpenCVViewer
from mobile_manipulation.utils.common import extract_scalars_from_info

def get_action_from_key(key, action_name):
    if "BaseArmGripperAction" in action_name:
        if key == "w":  # forward
            base_action = [1, 0]
        elif key == "s":  # backward
            base_action = [-1, 0]
        elif key == "a":  # turn left
            base_action = [0, 1]
        elif key == "d":  # turn right
            base_action = [0, -1]
        else:
            base_action = [0, 0]

        # End-effector is controlled
        if key == "i":
            arm_action = [1.0, 0.0, 0.0]
        elif key == "k":
            arm_action = [-1.0, 0.0, 0.0]
        elif key == "j":
            arm_action = [0.0, 1.0, 0.0]
        elif key == "l":
            arm_action = [0.0, -1.0, 0.0]
        elif key == "u":
            arm_action = [0.0, 0.0, 1.0]
        elif key == "o":
            arm_action = [0.0, 0.0, -1.0]
        else:
            arm_action = [0.0, 0.0, 0.0]

        if key == "f":  # grasp
            gripper_action = 1.0
        elif key == "g":  # release
            gripper_action = -1.0
        else:
            gripper_action = 0.0

        return {
            "action": "BaseArmGripperAction",
            "action_args": {
                "base_action": base_action,
                "arm_action": arm_action,
                "gripper_action": gripper_action,
            },
        }

def reset():
    obs = env.reset()
    info = {}
    print("episode_id", env.habitat_env.current_episode.episode_id)
    print("scene_id", env.habitat_env.current_episode.scene_id)
    return obs, info

if __name__ == "__main__":
    config = get_config('configs/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR.yaml')
    #config = get_config('configs/rearrange/skills/tidy_house/pick_v1_joint_SCR.yaml')
    env_cls = baseline_registry.get_env(config.ENV_NAME)
    env = env_cls(config)
    env = HabitatActionWrapperV1(env)

    env.seed(0)
    obs, info = reset()
    for k, v in obs.items():
        print(k, v.shape)
    viewer = OpenCVViewer(config.ENV_NAME)

    while True:
        metrics = extract_scalars_from_info(info)
        rendered_frame = env.render(info=metrics, overlay_info=False)
        key = viewer.imshow(rendered_frame)

        if key == "r":  # Press r to reset env
            obs, info = reset()
            continue

        action = get_action_from_key(key, config.RL.ACTION_NAME)

        obs, reward, done, info = env.step(action)
        print("step", env.habitat_env._elapsed_steps)
        print("action", action)
        print("reward", reward)
        print("info", info)

        if done:
            print("Done")
            obs, info = reset()
