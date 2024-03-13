from go1_gym.envs.go1.navigator import Navigator
from isaacgym import gymtorch, gymapi, gymutil
import torch
from params_proto import Meta
from typing import Union
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.utils.terrain import Terrain
import os
from isaacgym.torch_utils import *
import math
import pandas as pd

from go1_gym.envs.base.legged_robot_config import Cfg

class World(Navigator):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", locomtion_model_dir = "/common/home/ag2112/walk-these-ways/walk-these-ways/runs/gait-conditioned-agility/2023-11-03/train/210513.245978"):
        super().__init__(sim_device, headless, num_envs, prone,deploy,cfg,eval_cfg,initial_dynamics_dict,physics_engine, locomtion_model_dir=locomtion_model_dir)

        self.num_actions = 3
        self.total_reward_buffer = []
        self.goal_reward_buffer = []
        self.wall_penalty_buffer = []
        self.timeout_penalty_buffer = []
        self.iter = 0
        self.total_success = [0, 0]
        

        

    def update_goals(self, env_ids):
        self.init_root_states = self.root_states[self.num_actors_per_env * env_ids, :3]
        self.goals = self.root_states[self.anymal_actor_idxs, :3]
        self.goals[:, 0:1] += 3 * torch.ones((len(self.anymal_actor_idxs), 1)).to(self.goals.device)
        #print("Computed Goals!")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)


        # cube_asset_options = gymapi.AssetOptions()
        # cube_asset_options.use_mesh_materials = True
        # cube_asset_options.disable_gravity = True
        # cube_asset_options.fix_base_link = True
        # self.cube_asset = self.gym.load_asset(self.sim, asset_root, asset_file, cube_asset_options)
        # self.cube_num_dof = self.gym.get_asset_dof_count(self.cube_asset)
        # #self.cube_num_actuated_dof = self.num_actions
        # self.cube_num_bodies = self.gym.get_asset_rigid_body_count(self.cube_asset)
        # cube_dof_props_asset = self.gym.get_asset_dof_properties(self.cube_asset)
        # cube_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.cube_asset)
        



        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []
        self.anymal_actor_idxs = []
        self.wall_actor_idxs = []
        self.cube_actor_idxs = []
        self.cube_env_dim = []
        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()


        def make_wall(env_handle, start_pose, dimensions, env_num, wall_id):
            wall_asset_options = gymapi.AssetOptions()
            wall_asset_options.use_mesh_materials = True
            wall_asset_options.disable_gravity = True
            wall_asset_options.fix_base_link = True
            wall_asset = self.gym.create_box(self.sim, *dimensions, wall_asset_options)
            wall_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
            wall_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
            rigid_shape_props = self._process_rigid_shape_props(wall_rigid_shape_props, env_num)
            self.gym.set_asset_rigid_shape_properties(wall_asset, rigid_shape_props)
            wall_handle = self.gym.create_actor(env_handle, wall_asset, start_pose , f"wall_{wall_id}", env_num,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(wall_dof_props, env_num)
            self.gym.set_actor_dof_properties(env_handle, wall_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, wall_handle)
            body_props = self._process_rigid_body_props(body_props, env_num)
            self.gym.set_actor_rigid_body_properties(env_handle, wall_handle, body_props, recomputeInertia=True)
            return wall_handle
        
        def make_obstacle(env_handle, start_pose, dimensions, env_num):
            cube_asset_options = gymapi.AssetOptions()
            cube_asset_options.use_mesh_materials = True
            cube_asset_options.disable_gravity = True
            cube_asset_options.fix_base_link = True
            cube_asset = self.gym.create_box(self.sim, *dimensions, cube_asset_options)
            cube_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
            cube_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
            rigid_shape_props = self._process_rigid_shape_props(cube_rigid_shape_props, env_num)
            self.gym.set_asset_rigid_shape_properties(cube_asset, rigid_shape_props)
            cube_handle = self.gym.create_actor(env_handle, cube_asset, start_pose , "obstacle", env_num,
                                                  self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(cube_dof_props, env_num)
            self.gym.set_actor_dof_properties(env_handle, cube_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, cube_handle)
            body_props = self._process_rigid_body_props(body_props, env_num)
            self.gym.set_actor_rigid_body_properties(env_handle, cube_handle, body_props, recomputeInertia=True)
            self.cube_env_dim.append(dimensions)
            self.gym.set_rigid_body_color(env_handle, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0 ,0.0, 0.0))
            return cube_handle
        
        self.wall_handles = []
        self.cube_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            ## Adding Robot
            pos = self.env_origins[i].clone()
            x_min_range,x_max_range = 0.0,8.0
            y_min_range,y_max_range = -0.5,0.5
            
            # pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
            #                              device=self.device).squeeze(1)
            # pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
            #                              device=self.device).squeeze(1)
            # # pos[0:1] += torch_rand_float(x_min_range, x_max_range, (1, 1),
            #                              device=self.device).squeeze(1)
            # pos[1:2] += torch_rand_float(y_min_range, y_max_range, (1, 1),
            #                              device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)

            cube_offset = [0.8, 0.0, 1.0]
            cube_dim = [0.4, 0.3, 2.0]


            #randomize dimenstions
            cube_dim[1] += (torch_rand_float(0.0, 1.2, (1, 1), device=self.device).squeeze(1))

            #randomize offset based on y-dimension
            if cube_dim[1] <0.94:
                y_offset = (1 - 0.06) - (cube_dim[1]/2)
            else:
                y_offset = (1 - 0.06) - (cube_dim[1]/3)
            #y_offset_max = (pos[1] - 1 + 0.06) - (cube_dim[1]/2)
            cube_offset[1] += torch_rand_float(-y_offset, y_offset, (1, 1), device=self.device).squeeze(1).cpu().item()
            cube_offset[0] += torch_rand_float(0, 1.7, (1, 1), device=self.device).squeeze(1).cpu().item()
            cube_pose = gymapi.Transform()
            
            
            cofs = gymapi.Vec3(round(cube_offset[0], 2), round(cube_offset[1], 2), round(cube_offset[2], 2))
            cube_pose.p = start_pose.p + cofs 
            
            cube_handle = make_obstacle(env_handle, cube_pose, cube_dim, i)
            self.cube_handles.append(cube_handle)

            self.cube_actor_idxs.append(self.gym.get_actor_index(env_handle, cube_handle, gymapi.DOMAIN_SIM))
             



            ## Adding walls
            offset = [
                (1.2,1,1),
                (3.2,0,1),
                (1.2,-1,1),
                (-0.8,0,1),
            ]
            dimensions = [
                (4,0.12,2.0),
                (0.12,2.0,2.0),
                (4.0,0.12,2.0),
                (0.12,2.0,2.0),
            ]
            
            for j in range(4):
                ofs = gymapi.Vec3(*offset[j])
                dim = dimensions[j]

                tmp_pose = gymapi.Transform()
                tmp_pose.p = start_pose.p + ofs 

                wall_handle = make_wall(env_handle, tmp_pose, dim, i,j) 
                self.wall_handles.append(wall_handle)

                self.wall_actor_idxs.append(self.gym.get_actor_index(env_handle, wall_handle, gymapi.DOMAIN_SIM))


            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            self.anymal_actor_idxs.append(self.gym.get_actor_index(env_handle, anymal_handle, gymapi.DOMAIN_SIM))
        

        self.anymal_actor_idxs = torch.Tensor(self.anymal_actor_idxs).to(device=self.device,dtype=torch.long)
        self.wall_actor_idxs = torch.Tensor(self.wall_actor_idxs).to(device=self.device,dtype=torch.long)
        self.cube_actor_idxs = torch.Tensor(self.cube_actor_idxs).to(device=self.device,dtype=torch.long)
        self.num_actors_per_env = (len(self.actor_handles) + len(self.wall_handles)) // self.num_envs
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.extras = {}
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        anymal_env_ids = self.anymal_actor_idxs[env_ids].to(device=self.device)
        cube_env_ids = self.cube_actor_idxs[env_ids].to(device=self.device)
        ### Transform of robot which is the first actor of each environment
        # base position 
        if self.custom_origins:
            self.root_states[anymal_env_ids] = self.base_init_state
            self.root_states[anymal_env_ids, :3] += self.env_origins[env_ids]
            self.update_goals(env_ids)
            # self.root_states[self.num_actors_per_env * env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
            #                                                    cfg.terrain.x_init_range, (len(env_ids), 1),
            #                                                    device=self.device)
            # self.root_states[self.num_actors_per_env * env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
            #                                                    cfg.terrain.y_init_range, (len(env_ids), 1),
            #                                                    device=self.device)
            # x_min_range,x_max_range = 0.0,0.8
            # y_min_range,y_max_range = -0.5,0.5
            # self.root_states[anymal_env_ids, 0:1] += torch_rand_float(x_min_range, x_max_range, (1, 1),
            #                              device=self.device).squeeze(1)
            # self.root_states[anymal_env_ids, 1:2] += torch_rand_float(y_min_range, y_max_range, (1, 1),
            #                              device=self.device).squeeze(1)

            # self.root_states[anymal_env_ids, 0] += cfg.terrain.x_init_offset
            # self.root_states[anymal_env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[anymal_env_ids] = self.base_init_state
            self.root_states[anymal_env_ids, :3] += self.env_origins[env_ids]
            self.update_goals(env_ids)
        
        
        # base yaws
        #init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
        #                             cfg.terrain.yaw_init_range, (len(env_ids), 1),
        #                             device=self.device)
        #quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        
        #self.root_states[env_ids, 3:7] = torch.tensor(quat)

        # base velocities
        self.root_states[anymal_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        #env_ids_int32 = torch.arange(self.root_states.shape[0], dtype=torch.int32, device=env_ids.device)
        anymal_env_ids_int32 = anymal_env_ids.to(dtype=torch.int32)
        
        status = self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(anymal_env_ids_int32), len(anymal_env_ids_int32))
        
        cube_origins = self.env_origins[env_ids].clone()
        cube_env_dim_torch = torch.tensor(self.cube_env_dim, device=self.device)
        cube_dimensions = cube_env_dim_torch[env_ids].clone()
        
        for idx in range(len(cube_origins)):
            cube_offset = [0.8, 0.0, 1.0]
            y_dim = cube_dimensions[idx, 1]
            if y_dim < 0.94:
                y_offset = (1 - 0.06) - (y_dim / 2)
            else:
                y_offset = (1 - 0.06) - (y_dim / 3)

            cube_offset[1] += torch_rand_float(-y_offset, y_offset, (1, 1), device=self.device).squeeze(1).cpu().item()
            cube_offset[0] += torch_rand_float(0, 1.7, (1, 1), device=self.device).squeeze(1).cpu().item()
            cube_origins[idx] += torch.tensor(cube_offset, device=self.device)
        
        self.root_states[cube_env_ids, :3] = cube_origins

        cube_env_ids_int32 = cube_env_ids.to(dtype=torch.int32)
        
        status = self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(cube_env_ids_int32), len(cube_env_ids_int32))

        if not status:
            raise SystemError("Could not set necessary transforms")
        if cfg.env.record_video:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by + 1.0, bz + 6.0),
                                         gymapi.Vec3(bx, by, bz))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []
    
    def compute_reward(self):
        self.rew_buf[:] = 0
        env_ids = torch.arange(self.num_envs)
        anymal_env_ids = self.anymal_actor_idxs[env_ids].to(device=self.device)
        robot_pos = self.root_states[anymal_env_ids, 0:2]
        wall_env_ids = self.wall_actor_idxs.to(device=self.device)
        obs_env_ids = self.cube_actor_idxs.to(device = self.device)
        goal_pos = self.goals[:, 0:2]
        wall_pos = self.root_states[wall_env_ids, 0:2]
        obs_pos = self.root_states[obs_env_ids, 0:2]

        GOAL_REWARD = 10
        WALL_PENALTY = 0.001
        OBS_PENALTY = 0.001
        TIMEOUT_PENALTY = 0.001
        SCALING_FACTOR = 10
        goal_rew = []
        wall_rew = []
        obs_rew = []
        total_rew = []
        timeout_rew = []
        
        for i in range(self.num_envs):
            reward = -10
            dist_to_goal = torch.norm(robot_pos[i] - goal_pos[i])
            goal_reward_temp = 0
            if dist_to_goal < 0.2:
                goal_reward_temp += GOAL_REWARD * SCALING_FACTOR * 50
                self.total_success[0] += 1
                self.reset_buf[i] = True
            else:
                goal_reward_temp += GOAL_REWARD * SCALING_FACTOR / (dist_to_goal.cpu().item())
            
            goal_rew.append(goal_reward_temp)
            reward += goal_reward_temp

            wall_distance = []
            for j in range(4):
                dist = torch.norm(robot_pos[i] - wall_pos[(i*4)+j])
                wall_distance.append(dist)
            
            wall_distance = torch.tensor(wall_distance)
            min_wall_distance = torch.min(wall_distance)

            if min_wall_distance < 0.3:
                wall_penalty = WALL_PENALTY / min_wall_distance.cpu().item()
                reward-=wall_penalty
                wall_rew.append(wall_penalty)

            obs_distance = []
            obs_dist = torch.norm(robot_pos[i] - obs_pos[i])
            obs_distance.append(obs_dist)
            
            obs_distance = torch.tensor(obs_distance)

            if obs_distance < 0.3:
                obs_penalty = OBS_PENALTY/ obs_distance.cpu().item()
                reward-=obs_penalty
                obs_rew.append(obs_penalty)

            if self.time_out_buf[i]: 
                reward -= TIMEOUT_PENALTY
                timeout_rew.append(TIMEOUT_PENALTY)
                
            self.rew_buf[i] = reward
        
        contact_forces_temp = torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)
        
        self.rew_buf -= (contact_forces_temp)
        self.reset_buf |= contact_forces_temp > 0.1

        total_rew = self.rew_buf.cpu().tolist()
        self.goal_reward_buffer.append(goal_rew)
        if len(timeout_rew) > 0:
            self.timeout_penalty_buffer.append(timeout_rew)

        self.total_reward_buffer.append(total_rew)
        self.total_success[1] += self.num_envs
        
        

        if len(self.total_reward_buffer) == 30000:
            self.log_reward('/common/home/ag2112/walk-these-ways/walk-these-ways/rewards/storage')

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf = self.time_out_buf
        env_ids = torch.arange(self.num_envs)
        self.reset_buf |= (self.root_states[self.anymal_actor_idxs, 0] >= self.goals[:, 0])

    def _prepare_reward_function(self):
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in ["goal_reward","wall_penalty","timeout_penalty","total"]
        }

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        cfg.env.episode_length_s = 750 * self.dt
        super()._parse_cfg(cfg)

    
    def log_reward(self, filePath):
        # Create unique file names for each type of reward
        total_reward_file_name = f'{filePath}_total_reward_{self.iter}.csv'
        goal_reward_file_name = f'{filePath}_goal_reward_{self.iter}.csv'
        wall_penalty_file_name = f'{filePath}_wall_penalty_{self.iter}.csv'
        timeout_penalty_file_name = f'{filePath}_timeout_penalty_{self.iter}.csv'
        total_success_file_name = f'{filePath}_total_successes_{self.iter}.csv'

        # Create DataFrames from NumPy arrays
        total_reward_df = pd.DataFrame(self.total_reward_buffer)
        goal_reward_df = pd.DataFrame(self.goal_reward_buffer)
        wall_penalty_df = pd.DataFrame(self.wall_penalty_buffer)
        timeout_penalty_df = pd.DataFrame(self.timeout_penalty_buffer)
        total_success_df = pd.DataFrame(self.total_success)

        # Write the updated DataFrames to the CSV files
        total_reward_df.to_csv(total_reward_file_name, index=False, header=None, float_format='%.4f')
        goal_reward_df.to_csv(goal_reward_file_name, index=False, header=None, float_format='%.4f')
        wall_penalty_df.to_csv(wall_penalty_file_name, index=False, header=None, float_format='%.4f')
        timeout_penalty_df.to_csv(timeout_penalty_file_name, index=False, header=None, float_format='%.4f')
        total_success_df.to_csv(total_success_file_name, index=False, header=None, float_format='%.1f')

        # Clear the buffers
        self.total_reward_buffer = []
        self.goal_reward_buffer = []
        self.wall_penalty_buffer = []
        self.timeout_penalty_buffer = []
        self.iter += 1
