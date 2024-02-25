
class Config():
    def __init__(self):
        self.frame_l = 96 # the length of frames
        self.joint_n = 33 # the number of joints
        self.joint_d = 2 # the dimension of joints
        self.clc_num = 10 # the number of class
        self.feat_d = 528
        self.filters = 64
        self.nd = 60