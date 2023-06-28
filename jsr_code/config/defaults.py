from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_CPU = 4     
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.GPU_IDS = [0]                   # which gpus to use for training - list of int, e.g. [0, 1]
_C.SYSTEM.RNG_SEED = 42

_C.MODEL = CN()
#_C.MODEL.NET = "DeepLabRecon"                    # available networks from net.models.py file
_C.MODEL.NET = "DeepLabReconFuseSimpleTrain"                    # available networks from net.models.py file
_C.MODEL.BACKBONE = "resnet"                # choices: ['resnet', 'xception', 'drn', 'mobilenet']
_C.MODEL.OUT_STRIDE = 16                    # deeplab output stride
_C.MODEL.SYNC_BN = None                     # whether to use sync bn (for multi-gpu), None == Auto detect
_C.MODEL.FREEZE_BN = False                 

_C.MODEL.RECONSTRUCTION = CN()
_C.MODEL.RECONSTRUCTION.LATENT_DIM = 4      # number of channels of latent space
#_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210209_112729_903607/checkpoints/checkpoint-best.pth" #resnet 51.6
_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "jsr_code/checkpoints/checkpoint-best.pth"  #resnet 66.1
# _C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_213340_619898/checkpoints/checkpoint-best.pth" #mobilenet 61.2
#_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_214758_686017/checkpoints/checkpoint-best.pth" #xception 50.3
_C.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS = 19  # 19 for cityscapes
_C.MODEL.RECONSTRUCTION.SKIP_CONN = False 
_C.MODEL.RECONSTRUCTION.SKIP_CONN_DIM = 32 

_C.LOSS = CN()
#_C.LOSS.TYPE = "ReconstructionAnomalyLoss"           # available losses from net.loss.py
#_C.LOSS.TYPE = "ReconstructionAnomalyLossFuseSimple"           # available losses from net.loss.py
_C.LOSS.TYPE = "ReconstructionAnomalyLossFuseTrainAux"           # available losses from net.loss.py
_C.LOSS.IGNORE_LABEL = 255
_C.LOSS.SIZE_AVG = True
_C.LOSS.BATCH_AVG = True 

_C.EXPERIMENT= CN()
_C.EXPERIMENT.NAME = None                   # None == Auto name from date and time 
_C.EXPERIMENT.OUT_DIR = "/ssd/temporary/vojirtom/code_temp/sod/training/mcsegm/" 
_C.EXPERIMENT.EPOCHS = 200                  # number of training epochs
_C.EXPERIMENT.START_EPOCH = 0
_C.EXPERIMENT.USE_BALANCED_WEIGHTS = False
_C.EXPERIMENT.RESUME_CHECKPOINT = None      # path to resume file (stored checkpoint)
_C.EXPERIMENT.EVAL_INTERVAL = 1             # eval every X epoch
_C.EXPERIMENT.EVAL_METRIC = "AnomalyEvaluator" # available evaluation metrics from utils.metrics.py file

_C.INPUT = CN()
_C.INPUT.BASE_SIZE = 896 
_C.INPUT.CROP_SIZE = 896 
_C.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.BATCH_SIZE_TRAIN = None            # None = Auto set based on training dataset
_C.INPUT.BATCH_SIZE_TEST = None             # None = Auto set based on training batch size

_C.AUG = CN()
_C.AUG.RANDOM_CROP_PROB = 0.5               # prob that random polygon (anomaly) will be cut from image vs. random noise
_C.AUG.SCALE_MIN = 0.5
_C.AUG.SCALE_MAX = 2.0
_C.AUG.COLOR_AUG = 0.25

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.LR_SCHEDULER = "poly"          # choices: ['poly', 'step', 'cos']
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.NESTEROV = False

_C.DATASET = CN()
_C.DATASET.TRAIN = "cityscapes_2class"      # choices: ['cityscapes'],
_C.DATASET.VAL = "LaF"                      # choices: ['cityscapes'],
_C.DATASET.TEST = "LaF"                     # choices: ['LaF'],
_C.DATASET.FT = False                       # flag if we are finetuning 



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

