from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# GCAN model options
__C.GCAN = edict()
__C.GCAN.FEATURE_CHANNEL = 512
__C.GCAN.NODE_FEATURE_DIM = 1024
__C.GCAN.NODE_HIDDEN_SIZE = [512]
__C.GCAN.SK_ITER_NUM = 20
__C.GCAN.SK_EPSILON = 1.0e-10
__C.GCAN.SK_TAU = 0.005
__C.GCAN.CROSS_ITER = False
__C.GCAN.CROSS_ITER_NUM = 1