from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim import Ranger

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2_new', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', 'irseg_msv', 'irseg_samcl','VT','VT_Extended_TrainSet','VT_Extended_TrainSet_4000', 'RGBD_train','MFNet']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
    
    elif cfg['dataset'] == 'MFNet':
        from .datasets.irseg_samcl import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
    
    elif cfg['dataset'] == 'irseg_samcl':
        from .datasets.irseg_samcl import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
        
    elif cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PSTSeg
        return PSTSeg(cfg, mode='train'),PSTSeg(cfg, mode='test') ,PSTSeg(cfg, mode='test')
    
    elif cfg['dataset'] == 'VT':
        from .datasets.VT import VT8211000
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return VT8211000(cfg, mode='train'), VT8211000(cfg, mode='test'),VT8211000(cfg, mode='test')
    
    elif cfg['dataset'] == 'VT_Extended_TrainSet':
        from .datasets.VT_Extended_TrainSet import VT8211000
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return VT8211000(cfg, mode='train'), VT8211000(cfg, mode='test'),VT8211000(cfg, mode='test')
    
    elif cfg['dataset'] == 'VT_Extended_TrainSet_4000':
        from .datasets.VT_Extended_TrainSet_4000 import VT8211000
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return VT8211000(cfg, mode='train'), VT8211000(cfg, mode='test'),VT8211000(cfg, mode='test')
    
    elif cfg['dataset'] == 'RGBD_train':
        from .datasets.RGBD_train import RGBD
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return RGBD(cfg, mode='train'), RGBD(cfg, mode='test'),RGBD(cfg, mode='test')
    


def get_model(cfg):

    if cfg['model_name'] == 'EGFNet':
        from .models.EGFNet import EGFNet
        return EGFNet(n_classes=cfg['n_classes'])
     
    elif cfg['model_name']== 'LASNet_base':
        from .models.LASNet_base import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_RGB':
        from .models.LASNet_base_RGB import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_RGBTD':
        from .models.LASNet_base_RGBTD import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_new_contrastive':
        from .models.LASNet_base_new_contrastive import LASNet
        return LASNet(n_classes=cfg['n_classes'])

    elif cfg['model_name']== 'LASNet_base_CML_CCL':
        from .models.LASNet_base_CML_CCL import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_CWCL':
        from .models.LASNet_base_CWCL import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_CWCL_res2net101':
        from .models.LASNet_base_CWCL_res2net101 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_CWCL_resnet50':
        from .models.LASNet_base_CWCL_resnet50 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_CWCL_resnet152':
        from .models.LASNet_base_CWCL_resnet152 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_CWCL_res2net50_vgg16':
        from .models.LASNet_base_CWCL_res2net50_vgg16 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
  
    elif cfg['model_name']== 'LASNet_base_CWCL_SDCL':
        from .models.LASNet_base_CWCL_SDCL import LASNet
        return LASNet(n_classes=cfg['n_classes'])

    elif cfg['model_name']== 'LASNet_base_4ch':
        from .models.LASNet_base_4ch import LASNet
        return LASNet(n_classes=cfg['n_classes'])

    elif cfg['model_name']== 'LASNet_pretrain':
        from .models.LASNet_pretrain import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_samcl':
        from .models.LASNet_base_samcl import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_v1':
        from .models.LASNet_base_v1 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_pst900':
        from .models.LASNet_base_pst900 import LASNet
        return LASNet(n_classes=cfg['n_classes'])
        
    elif cfg['model_name']== 'LASNet_base_pst900_CWCL':
        from .models.LASNet_base_pst900_CWCL import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_pst900_RGB':
        from .models.LASNet_base_pst900_RGB import LASNet
        return LASNet(n_classes=cfg['n_classes'])
        
    else:
        from .models.LASNet import LASNet
        return LASNet(n_classes=cfg['n_classes'])
        #return LASNet(n_classes=5)
        
