# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
from .stereobase_gru import StereoBase as StereoBaseGRU

__all__ = {
    'StereoBaseGRU': StereoBaseGRU,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, wdb_writer=None):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model, wdb_writer=wdb_writer)
