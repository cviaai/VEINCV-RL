import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import numpy.ma as ma

import gym
from gym import spaces
from gym.envs.toy_text import discrete

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env

from pathlib import Path
import os, sys, inspect
import random


def load_image_mask(path, img_path, msk_path):
    """ Load image and mask (in final prototype will be received from previous step in pipeline)

    Parameters:
        path: relative path to folder with image and mask files
        img_path: image file name (with extension)
        msk_path: mask file name (with extension)

    Returns:
        image: image loaded
        mask: mask loaded
    """

    image = cv2.cvtColor(cv2.imread(os.path.join(path, img_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(path, msk_path))
    thrshd = 100  ### to delete artifacts
    mask[mask > thrshd] = 255
    mask[mask <= thrshd] = 0
    return image, mask


def analyze_image(image, mask):
    """ Given image and mask calc the "projection alignment" as std over mean in our ROI

    Parameters:
        image: image (int n*m matrix). grayscale
        mask: image (int n*m matrix). grayscale

    Returns:
        align: metric for alignment
    """
    vett = np.array(np.nonzero(mask))
    roi = [vett[0].min(), vett[0].max(), vett[1].min(), vett[1].max()]
    target_cut = mask[roi[0]:roi[1], roi[2]:roi[3]]
    mx = ma.masked_array(image[roi[0]:roi[1], roi[2]:roi[3]], [target_cut == 0])

    align = np.round(mx.std(ddof=1) / mx.mean(), 4)
    return align


def mask_red(mask):
    """ Given mask generate a red mask over white background

    Parameters:
        mask: image (int n*m matrix).

    Returns:
        red_mask: mask with red colored veins
    """
    vett = np.array(np.nonzero(mask))
    roi = [vett[0].min(), vett[0].max(), vett[1].min(), vett[1].max()]
    cut = mask[roi[0]:roi[1], roi[2]:roi[3]]
    red_mask = np.zeros((cut.shape[0], cut.shape[1], 3), dtype=np.uint8)
    red_mask.fill(255)

    if (cut[:, :].shape[2] == 1):
        red_mask[..., 1] -= cut[:, :]
        red_mask[..., 0] -= cut[:, :]
    else:
        red_mask[..., 1] -= cut[:, :, 1]
        red_mask[..., 0] -= cut[:, :, 0]

    return red_mask


def rotate_scale(image, angle=0, scale=1):
    """ rotate and scale an image, for rotation add also white background

    Parameters:
        image: image (int n*m matrix).
        angle: angle of rotation in degrees
        scale: scaling value

    Returns:
        ret: image rotated and or scaled
    """

    # grab the dimensions of the image and then determine the
    # center
    img = image.copy()
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))


def sim_projection(image, mask, rows, cols, angle, scale=1.0):
    """ merge the mask with the image, (simulation of projector use)

    Parameters:
        image: image (int n*m*c matrix).
        mask: image (int n*m matrix).
        rows: int row position (inside image) where to merge the mask
        cols: int column position (inside image) where to merge the mask
        angle: angle of rotation in degrees
        scale: scaling value

    Returns:
        merge: image, (int n*m*c) image with mask transformed and merged onto
    """
    merge = image.copy()

    rows = int(rows)
    cols = int(cols)

    # rotation
    rotated = rotate_scale(mask, angle, scale)

    ### for recalculation of vertices of bounding box
    center_r = int((rotated.shape[0] - mask.shape[0]) / 2)
    center_c = int((rotated.shape[1] - mask.shape[1]) / 2)

    # coordinates where to position the transformed mask
    prj_crds = [rows - center_r, rows - center_r + rotated.shape[0], cols - center_c, cols - center_c + rotated.shape[1]]

    # merge the 2 images
    img_overlap = cv2.addWeighted(merge[prj_crds[0]:prj_crds[1], prj_crds[2]:prj_crds[3]], 0.8, rotated, 0.5, 0)
    merge[prj_crds[0]:prj_crds[1], prj_crds[2]:prj_crds[3]] = img_overlap

    # return
    return merge

    ## Complete Version (translation, rotation and scaling)


class ProjectionEnv(gym.Env):
    """
    Custom Environment that follows gym interface.

    """

    metadata = {'render.modes': ['console', 'rgb_array']}
    # Define constants for clearer code ### for 1 pixel
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    CLOCK = 4  ### for clock rotation
    COUNT = 5  ### for counterclock rotation
    INCR = 6
    DECR = 7

    nA = 8  ### number of actions

    def __init__(self):
        super(ProjectionEnv, self).__init__()

        self.max_steps = 100
        self.time = 0

        main_path = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(ProjectionEnv)), "..", "images"))

        img_path = "screenshot_Y.jpg"
        msk_path = 'mask.jpg'

        # class attributes, image from
        self.image, self.mask = load_image_mask(main_path, img_path, msk_path)
        self.mask_prj = mask_red(self.mask)

        self.image_rows = self.image.shape[0]
        self.image_cols = self.image.shape[1]

        self.mask_rows = self.mask_prj.shape[0]
        self.mask_cols = self.mask_prj.shape[1]

        ### maximum displacement (for shifting)
        self.rows = 51
        self.cols = 51

        self.max_rot = 10
        self.max_scale = 0.10


        # target real position (later remove and check with std/mean)
        ### real one for now
        self.target_real_row = 149
        self.target_real_col = 515

        # target relative positions (later remove and check with std/mean)
        self.target_row = 3
        self.target_col = 10
        self.target_rot = 0

        # agent real position
        ### how far are we from real (final) coordinates (from grid to grid)
        self.agent_real_row = self.target_real_row - self.target_row
        self.agent_real_col = self.target_real_col - self.target_col


        # Define action and observation space
        # They must be gym.spaces objects

        # Action space
        self.action_space = spaces.Discrete(self.nA)

        # The observation will be the coordinate of the agent
        ### coordinates where actualy it on grid
        self.low_state = np.array([0, 0, -self.max_rot, 1 - self.max_scale, 0],
                                  dtype=np.float32)  ### not less than 0 - can be fixed
        self.high_state = np.array([self.rows, self.cols, self.max_rot, 1 + self.max_scale, 1],
                                   dtype=np.float32)  ### for our case may be fine, but need to chack for value "1"
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.reset()

    def reset(self):
        self.time = 0

        self.contrast = [0]

        self.time = 0

        self.done = False


        # agent relative positions start, initial state of projection
        ### it is affine transformed already by these parameters (would like to reach it)
        ### just for start, then it will be randomized
        self.rel_row = int(self.rows / 2)
        self.rel_col = int(self.cols / 2)
        self.rel_rot = 0
        self.rel_scale = 1

        # Initialize the agent relative pos ###(rel = relative)
        self.agent_rel_pos = np.array([self.rel_row, self.rel_col])


        # Initialize the agent relative pos
        self.agent_rel_pos = np.array([self.rel_row, self.rel_col, self.rel_rot, self.rel_scale])

        # Initialize the agent real position        
        self.agent_real_pos = np.array([self.agent_real_row, self.agent_real_col, self.rel_rot, self.rel_scale])


        # Image with mask virtually projected over "camera" image
        self.merge = sim_projection(self.image.copy(), self.mask_prj.copy(), self.agent_real_pos[0],
                                    self.agent_real_pos[1], self.agent_real_pos[2], self.agent_real_pos[3])

        # calc contrast over ROI
        self.contrast = np.round(analyze_image(self.merge, self.mask), 4)

        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.state = np.array(
            [self.agent_rel_pos[0], self.agent_rel_pos[1], self.agent_rel_pos[2], self.agent_rel_pos[3],
             self.contrast])

        return self.state

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        bump = False  ### invisible wall = cannot move
        done = False

        if action == self.UP:
            if self.agent_rel_pos[0] > 0:
                self.agent_rel_pos[0] -= 1
                self.agent_real_pos[0] -= 1
            else:
                bump = True
        elif action == self.DOWN:
            if self.agent_rel_pos[0] < self.rows:
                self.agent_rel_pos[0] += 1
                self.agent_real_pos[0] += 1
            else:
                bump = True
        elif action == self.LEFT:
            if self.agent_rel_pos[1] > 0:
                self.agent_rel_pos[1] -= 1
                self.agent_real_pos[1] -= 1
            else:
                bump = True
        elif action == self.RIGHT:
            if self.agent_rel_pos[1] < self.cols:
                self.agent_rel_pos[1] += 1
                self.agent_real_pos[1] += 1
            else:
                bump = True
        elif action == self.COUNT:
            if self.agent_rel_pos[2] < self.max_rot:
                self.agent_rel_pos[2] += 1
                self.agent_real_pos[2] += 1
            else:
                bump = True
        elif action == self.CLOCK:
            if self.agent_rel_pos[2] > -self.max_rot:
                self.agent_rel_pos[2] -= 1
                self.agent_real_pos[2] -= 1
            else:
                bump = True
        elif action == self.DECR:
            if self.agent_rel_pos[3] > 1.0 - self.max_scale:
                self.agent_rel_pos[3] -= 0.01
                self.agent_real_pos[3] -= 0.01
            else:
                bump = True
        elif action == self.INCR:
            if self.agent_rel_pos[3] < 1.0 + self.max_scale:
                self.agent_rel_pos[3] += 0.01
                self.agent_real_pos[3] += 0.01
            else:
                bump = True
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # simulate projection over video    
        self.merge = sim_projection(self.image.copy(), self.mask_prj.copy(), self.agent_real_pos[0],
                                    self.agent_real_pos[1], self.agent_real_pos[2], self.agent_real_pos[3])

        # calc contrast over ROI
        self.contrast = np.round(analyze_image(self.merge, self.mask), 4)

        if self.time < self.max_steps:
            self.time += 1
        else:
            done = True
        ### to reach the goal faster (for all motions != bump)
        ### also for maximizing contrast
        reward = self.contrast - 1
        ### for wall move
        if bump:
            reward = -1

        if (self.agent_real_pos[0] == self.target_real_row and self.agent_real_pos[1] == self.target_real_col and
                self.agent_real_pos[2] == self.target_rot and self.agent_real_pos[3] == 1.0):
            done = True
            reward = 10

        # Optionally we can pass additional info, we are not using that for now
        ### 
        info = {'xAG': self.agent_real_pos[0], 'xTARG': self.target_real_row,
                'jAG': self.agent_real_pos[1], 'jTARG': self.target_real_col,
                'rotAG': self.agent_real_pos[2], "scale=": self.agent_real_pos[3]}

        state = np.array([self.agent_rel_pos[0], self.agent_rel_pos[1], self.agent_rel_pos[2], self.agent_rel_pos[3],
                          self.contrast])
        return state, reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return (self.merge)

    def close(self):
        pass
