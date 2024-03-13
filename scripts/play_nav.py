import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.go1.world import World
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def load_policy_nav(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os

    def policy_nav(obs, info={}):
        i = 0
        action = body.forward(obs["nav_obs_history"].to('cpu'))
        return action

    return policy_nav

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load('/common/home/ag2112/walk-these-ways/walk-these-ways/tmp/legged_data/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def saveToVideo(frames, output_video_name, frame_rate=25.0, codec='mp4v'):
    # Get the shape of the first frame to determine video dimensions
    frame_height, frame_width, _ = frames[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_name, fourcc, frame_rate, (frame_width, frame_height))

    # Loop through the frames and write each frame to the video
    for frame in frames:
        # Convert from RGB to BGR (OpenCV uses BGR format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the VideoWriter
    out.release()

def load_env(label, label_nav, headless=False):
    # dirs = glob.glob(f"../runs/{label}/*")
    # logdir = sorted(dirs)[0]
    # logdir = f"{label}"

    with open(label + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.env.record_video = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper_nav import HistoryWrapper

    #env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = World(sim_device='cuda:0',headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_nav.actor_critic_nav import ActorCritic_Nav

    policy_nav = load_policy_nav(label_nav)
    policy = load_policy(label)
    return env, policy, policy_nav

def dog_walk(env,policy, policy_nav, obs, num_eval_steps = 750):
    # num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, -0.5]}
    
    

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_vels = np.zeros((num_eval_steps,3))
    i = 0
    done = False
    while not done and i < num_eval_steps:
        with torch.no_grad():
            actions = policy(obs)
            actions_nav = policy_nav(obs)
        #getting the automated velocities from the policy
        
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = actions_nav[0]
        actions_nav[0] = torch.tensor([0.25, 0.0, 0.0])
        
        obs, rew, done, info = env.step(actions_nav[0])

        
        measured_vels[i,:] = env.base_lin_vel[0, :].cpu()
        measured_vels[i,2] = env.base_ang_vel[0, 2].cpu()
        i+=1
    return measured_vels

def play_go1(headless=False):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    
    label_nav = "/common/home/ag2112/walk-these-ways/walk-these-ways/ag2112/scratch/2023/12-06/215436"

    label = "/common/home/ag2112/walk-these-ways/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545"

    env, policy, policy_nav = load_env(label, label_nav, headless=headless)
    env.record_video = True
    env.start_recording()

    obs = env.reset()
    #obs_vel = dog_walk(env, policy, policy_nav, obs)

    num_eval_steps = 550
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, -0.5]}
    
    

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_vels = np.zeros((num_eval_steps,3))
    i = 0
    done = False
    while not done and i < num_eval_steps:
        with torch.no_grad():
            actions = policy(obs)
            actions_nav = policy_nav(obs)
        #getting the automated velocities from the policy
        
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = actions_nav[0]
        #actions_nav[0] = torch.tensor([0.25, 0.0, 0.0])
        
        obs, rew, done, info = env.step(actions_nav[0])
        measured_vels[i,:] = env.base_lin_vel[0, :].cpu()
        measured_vels[i,2] = env.base_ang_vel[0, 2].cpu()
        #print('Contact Forces: ', env.contact_forces[:, env.penalised_contact_indices, :])
        i+=1


    frames = env.video_frames
    env.pause_recording()
    print(len(frames))
    record_frames(frames)
    #print(f'measured velocity:{measured_vels}')

def record_frames(frames):
    for i,x in enumerate(frames):
        plt.imsave("/common/home/ag2112/walk-these-ways/walk-these-ways/scripts/videos/frame-"+str(i)+".jpg",x)
    
    images = [img for img in os.listdir("/common/home/ag2112/walk-these-ways/walk-these-ways/scripts/videos") if img.endswith(".jpg")]
    
    def getNum(name):
        num = name.split('-')[1].split('.')[0]
        return int(num)
    
    images.sort(key=getNum)
    
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter("video2.avi",0, 20, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join("/common/home/ag2112/walk-these-ways/walk-these-ways/scripts/videos", image)))

    cv2.destroyAllWindows()
    video.release()    


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)