import os
import numpy as np
import random
import torch
import argparse
from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_eval_rrs_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference_rrs_visualize


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # evaluation details
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/vit_clipreid_288x144_v2.yml", help="path to config file", type=str
    )
    parser.add_argument("--sort", default="ascending", type=str, help="Sorting order for visualization")
    parser.add_argument("--num_vis", default=10, type=int, help="Number of images to visualize")
    parser.add_argument("--rank_vis", default=10, type=int, help="Number of ranks to visualize")
    parser.add_argument("--vis_output_dir", default="./vis_results", type=str, help="Output directory for visualization results")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # create validation datasets
    val_loader, num_query, num_classes, camera_num, view_num = make_eval_rrs_dataloader(cfg)

    # create and load the model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param("logs/gradient_centralization/checkpoint_ep.pth.tar")

    # perform the evaluation
    do_inference_rrs_visualize(
        cfg,
        model,
        val_loader,
        num_query,
        visualize=True,
        sort=args.sort,
        num_vis=args.num_vis,
        rank_vis=args.rank_vis,
        output_dir=args.vis_output_dir
    )
