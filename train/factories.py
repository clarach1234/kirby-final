from utils_pytorch import UniversalFactory
import models
import models_v2
import losses



class ModelFactory(UniversalFactory):
    classes = [
        models.UNet,
        models.WNet,
        models.RRUNet,
        models.RRWNetAll,
        models.RRWNet,
        models.CMRRWNet,
        models_v2.CMRRWNetV2,
    ]


class LossesFactory(UniversalFactory):
    classes = [
        losses.BCE3Loss,
        losses.RRLoss,
        losses.OfficialScoreLoss,
    ]
