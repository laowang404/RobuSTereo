# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

from .models.casnet.trainer import Trainer as CasStereoTrainer
from .models.cfnet.trainer import Trainer as CFNetTrainer
# from .models.aanet.trainer import Trainer as AANetTrainer
from .models.coex.trainer import Trainer as CoExTrainer
from .models.fadnet.trainer import Trainer as FADNetTrainer
from .models.gwcnet.trainer import Trainer as GwcNetTrainer
from .models.igev.trainer import Trainer as IGEVTrainer
from .models.msnet.trainer import Trainer as MSNetTrainer
from .models.psmnet.trainer import Trainer as PSMNetTrainer
from .models.sttr.trainer import Trainer as STTRTrainer
from .models.lightstereo.trainer import Trainer as LightStereoTrainer
from .models.stereobase.trainer import Trainer as StereoBaseTrainer
from .models.dinobase.trainer import Trainer as DinoBaseTrainer
from .models.selective_igev.trainer import Trainer as SelectiveIGEVTrainer
from .models.nmrf.trainer import Trainer as NMRFTrainer

__all__ = {
    'STTR': STTRTrainer,
    'PSMNet': PSMNetTrainer,
    'MSNet2D': MSNetTrainer,
    'MSNet3D': MSNetTrainer,
    'IGEV': IGEVTrainer,
    'GwcNet': GwcNetTrainer,
    'FADNet': FADNetTrainer,
    'CoExNet': CoExTrainer,
    # 'AANet': AANetTrainer,
    'CFNet': CFNetTrainer,
    'CasGwcNet': CasStereoTrainer,
    'CasPSMNet': CasStereoTrainer,
    'LightStereo': LightStereoTrainer,
    'StereoBaseGRU': StereoBaseTrainer,
    'DinoBase': DinoBaseTrainer,
    'SelectiveIGEV':SelectiveIGEVTrainer,
    'NMRF': NMRFTrainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer, wdb_writer=None):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer, wdb_writer=wdb_writer)
    return trainer
