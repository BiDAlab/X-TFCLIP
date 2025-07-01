import torch.nn.functional as F
from .online_label_smooth import OnlineLabelSmoothing
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

class make_loss:
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.num_classes = num_classes
        self.sampler = cfg.DATALOADER.SAMPLER
        self.feat_dim = 2048

        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=self.feat_dim, use_gpu=True)

        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.NO_MARGIN:
                self.triplet = TripletLoss()
                print("using soft triplet loss for training")
            else:
                self.triplet = TripletLoss(cfg.SOLVER.MARGIN)
                print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
        else:
            raise ValueError(f"Expected METRIC_LOSS_TYPE to contain 'triplet', but got {cfg.MODEL.METRIC_LOSS_TYPE}")

        self.label_smoothing_on = cfg.MODEL.IF_LABELSMOOTH == 'on'
        self.xent = OnlineLabelSmoothing(n_classes=num_classes).cuda() if self.label_smoothing_on else None

    def compute_loss(self, score, feat, target, target_cam=None, i2tscore=None, isprint=False):
        if self.sampler == 'softmax':
            return F.cross_entropy(score, target)

        elif self.sampler == 'softmax_triplet':
            return self._compute_softmax_triplet_loss(score, feat, target, i2tscore, isprint)

        else:
            raise ValueError(f"Expected sampler to be softmax or softmax_triplet but got {self.sampler}")

    def _compute_softmax_triplet_loss(self, score, feat, target, i2tscore, isprint):
        cfg = self.cfg
        I2TLOSS = 0.0

        if self.label_smoothing_on:
            ID_LOSS = self._compute_loss_list(self.xent, score, target)
        else:
            ID_LOSS = self._compute_loss_list(F.cross_entropy, score, target)

        TRI_LOSS = self._compute_triplet_loss(feat, target)
        loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

        if i2tscore is not None:
            if self.label_smoothing_on:
                I2TLOSS = self._compute_loss_list(self.xent, i2tscore, target)
            else:
                I2TLOSS = self._compute_loss_list(F.cross_entropy, i2tscore, target)
            loss += cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS

        if isprint:
            print(f"Loss: {loss:.3f}, ID Loss: {ID_LOSS:.3f}, TRI Loss: {TRI_LOSS:.3f}, I2T Loss: {I2TLOSS:.3f}")

        return loss

    def _compute_loss_list(self, loss_fn, scores, target):
        if isinstance(scores, list):
            return sum([loss_fn(s, target) for s in scores])
        else:
            return loss_fn(scores, target)

    def _compute_triplet_loss(self, features, target):
        if isinstance(features, list):
            return sum([self.triplet(f, target)[0] for f in features])
        else:
            return self.triplet(features, target)[0]
