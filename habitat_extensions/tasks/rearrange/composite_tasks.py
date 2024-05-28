import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from .sim import RearrangeSim
from .task import RearrangeEpisode, RearrangeTask
from .task_utils import (
    check_start_state,
    compute_start_state,
    sample_random_start_state_v1,
    compute_region_goals_v1,
    sample_noisy_start_state,
)


@registry.register_task(name="TidyHouseTask-v0")
class TidyHouseTask(RearrangeTask):
    def initialize(self, episode: RearrangeEpisode):
        if self._config.get("FRIDGE_INIT", False):
            self._sim.set_fridge_state_by_motor(2.356)

        start_state = self.sample_start_state()
        if start_state is None:
            raise RuntimeError(
                "Episode {}: fail to sample a valid start state".format(
                    episode.episode_id
                )
            )
        self._sim.robot.base_pos = start_state[0]
        self._sim.robot.base_ori = start_state[1]
        self._sim.internal_step_by_time(0.1)

        # Cache start positions
        self.obj_start_pos = self._sim.get_rigid_objs_pos_dict()
        tgt_idx = self._config.get("TARGET_INDEX", 0)
        if tgt_idx == -1:
            tgt_idx = self.np_random.choice(len(self._sim.targets))
        self.set_target(tgt_idx)

    def set_target(self, index):
        self.tgt_idx = index
        self.tgt_obj, self.tgt_T = self._sim.get_target(self.tgt_idx)
        self.pick_goal = self.obj_start_pos[self.tgt_obj.handle]
        self.place_goal = np.array(self.tgt_T.translation, dtype=np.float32)
        self.nav_goal_pick = compute_start_state(self._sim, self.pick_goal)
        self.nav_goal_place = compute_start_state(self._sim, self.place_goal)

    def sample_start_state(self, max_trials=20, verbose=False):
        self._initialize_ee_pos()

        for i in range(max_trials):
            start_state = sample_random_start_state_v1(
                self._sim, max_trials=20, rng=self.np_random
            )
            if start_state is None:
                if verbose:
                    print("The goal is not navigable")
                continue
            is_valid = check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="nav",
                max_collision_force=0.0,
                verbose=verbose,
            )
            if is_valid:
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def render(self, mode):
        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)
        self._sim.visualize_arrow(
            "nav_goal_pick",
            self.nav_goal_pick[0],
            self.nav_goal_pick[1],
            scale=0.3,
        )
        self._sim.visualize_arrow(
            "nav_goal_place",
            self.nav_goal_place[0],
            self.nav_goal_place[1],
            scale=0.3,
        )
        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret


@registry.register_task(name="SetTableTask-v0")
class SetTableTask(RearrangeTask):
    def initialize(self, episode: RearrangeEpisode):
        start_state = self.sample_start_state()
        if start_state is None:
            raise RuntimeError(
                "Episode {}: fail to sample a valid start state".format(
                    episode.episode_id
                )
            )

        self._sim.robot.base_pos = start_state[0]
        self._sim.robot.base_ori = start_state[1]
        obj_start_pos = self._sim.get_rigid_objs_pos_dict()
        self._sim.internal_step_by_time(0.1)

        # Cache start positions
        self.obj_start_pos = self._sim.get_rigid_objs_pos_dict()
        tgt_idx = self._config.get("TARGET_INDEX", 0)
        self.set_target(tgt_idx)

        # -------------------------------------------------------------------------- #
        # Sanity check
        # -------------------------------------------------------------------------- #
        handle = self.tgt_obj.handle
        err = np.linalg.norm(
            obj_start_pos[handle] - self.obj_start_pos[handle]
        )
        if err > 0.05:
            logger.warning(
                "Episode {}({}): start pos err {}".format(
                    episode.episode_id, self.tgt_idx, err
                )
            )

    def set_target(self, index):
        self._set_target(index)
        self.pick_goal = self.obj_start_pos[self.tgt_obj.handle]
        self.place_goal = np.array(self.tgt_T.translation, dtype=np.float32)

        if self._has_target_in_fridge():
            self.marker_name = "fridge_push_point"
            self.marker = self._sim.markers[self.marker_name]
            # self.tgt_qpos = 0.0
            self.tgt_qpos = 1.57  # for mono RL only
        elif self._has_target_in_drawer():
            self.marker_name = "cab_push_point_{}".format(
                self.tgt_receptacle_info[1]
            )
            self.marker = self._sim.markers[self.marker_name]
            # self.tgt_qpos = 0.0
            self.tgt_qpos = 0.45  # for mono RL only
        else:
            raise NotImplementedError(index)

    def sample_start_state(self, max_trials=20, verbose=False):
        self._initialize_ee_pos()

        for i in range(max_trials):
            start_state = sample_random_start_state_v1(
                self._sim,
                max_trials=20,
                rng=self.np_random,
            )
            if start_state is None:
                if verbose:
                    print("The goal is not navigable")
                continue
            is_valid = check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="nav",
                max_collision_force=0.0,
                verbose=verbose,
            )
            if is_valid:
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def render(self, mode):
        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)
        self._sim.visualize_frame(
            self.marker_name, self.marker.transformation, scale=0.15
        )
        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret

@registry.register_task(name="TidyHouseTask-v1")
class TidyHouseTaskV1(TidyHouseTask):
    def initialize(self, episode: RearrangeEpisode):
        task = np.random.randint(2) # 0 for pick, 1 for place

        if self._config.get("FRIDGE_INIT", False):
            self._sim.set_fridge_state_by_motor(2.356)

        # Cache start positions
        self.obj_start_pos = self._sim.get_rigid_objs_pos_dict()
        tgt_idx = self._config.get("TARGET_INDEX", 0)
        if tgt_idx == -1:
            tgt_idx = self.np_random.choice(len(self._sim.targets))
        self.set_target(tgt_idx)

        start_state = self.sample_start_state(episode, task)

        if start_state is None:
            start_state = self.sample_start_state(episode, task, no_validation=True)
            logger.warning(
                "Episode {}({}): sample a start state without validation".format(
                    episode.episode_id, episode.scene_id
                )
            )

        self._sim.robot.base_pos = start_state[0]
        self._sim.robot.base_ori = start_state[1]
        if task == 1:
            self._sim.robot.open_gripper()
            self._sim.gripper.snap_to_obj(self.tgt_obj)
        self._sim.internal_step_by_time(0.1)

    def sample_start_state(self, episode, task, max_trials=20, verbose=False, no_validation=False):
        goal_pos = self.pick_goal if task == 0 else self.place_goal
        start_pos, _ = compute_start_state(self._sim, goal_pos)
        height = start_pos[1]
        T = mn.Matrix4.translation(goal_pos)
        start_positions = compute_region_goals_v1(
            self._sim,
            T,
            region=None,
            radius=2.0,
            height=height,
            max_radius=2.0,
            debug=False,
        )

        if start_positions is None or len(start_positions) == 0:
            logger.warning(
                "Episode {} ({}): Unable to find any navigable point around the {}-th target given the map.".format(
                    episode.episode_id, episode.scene_id, self.tgt_idx
                )
            )
            return None

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        for i in range(max_trials):
            # Avoid extreme end-effector positions by resampling each time
            self._initialize_ee_pos()

            ind = self.np_random.choice(len(start_positions))
            start_state = sample_noisy_start_state(
                self._sim,
                start_positions[ind],
                # pick_goal,
                goal_pos,  # Note we use goal specification here!
                pos_noise=pos_noise,
                ori_noise=ori_noise,
                pos_noise_thresh=2 * pos_noise,
                ori_noise_thresh=2 * ori_noise,
                max_trials=10,
                verbose=verbose,
                rng=self.np_random,
            )
            if start_state is None:
                continue

            if no_validation:
                return start_state

            if check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="pick",
                max_collision_force=0.0,
                verbose=verbose,
            ):
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state
