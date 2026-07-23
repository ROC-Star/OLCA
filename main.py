# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from mmengine import Config
from torch.utils import data
from tqdm import tqdm

import methods as model_zoo
from utils import io, ops, pipeline, pt_utils, py_utils, recorder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

logger.remove()
logger_format = "[<green>{time:YYYY-MM-DD HH:mm:ss} - {file}</>] <lvl>{message}</>"
logger.add(sys.stderr, level="DEBUG", format=logger_format)


class ImageTrainDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: dict, descriptions_json_path: str):
        super().__init__()
        self.input_hw = input_hw
        self.descriptions_json_path = descriptions_json_path

        # Load the description JSON file
        self.descriptions = self._load_descriptions()

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "train":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TR_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TR_MASK_DIR, unique_id + mask_suffix)
            depth_path = os.path.join(dataset_info.OVCamo_TR_DEPTH_DIR, unique_id + mask_suffix)

            # Check whether the description exists
            if unique_id not in self.descriptions:
                logger.warning(f"Description for {unique_id} was not found; using the default description")

            self.total_data_paths.append((class_name, image_path, mask_path, depth_path, unique_id))

        logger.info(f"[TrainSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")


        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ],
            # # Additional target: treat depth as a mask
            additional_targets={"depth": "mask"},
        )

    def _load_descriptions(self):
        """Load the description JSON file."""
        if not os.path.exists(self.descriptions_json_path):
            logger.warning(f"Description file does not exist: {self.descriptions_json_path}")
            return {}

        try:
            with open(self.descriptions_json_path, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            logger.info(f"Loaded {len(descriptions)} descriptions")
            return descriptions
        except Exception as e:
            logger.error(f"Failed to load description file: {e}")
            return {}

    def __getitem__(self, index):
        class_name, image_path, mask_path, depth_path, unique_id = self.total_data_paths[index]

        desc = self.descriptions.get(unique_id, {})
        foreground_desc = desc.get("foreground_description", f"A camouflaged {class_name}")
        background_desc = desc.get("background_description", "The surrounding environment")

        # Read image, mask, and depth map
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        mask = (mask * 255).astype(np.uint8)
        depth = io.read_gray_array(depth_path, to_normalize=True)
        depth = (depth * 255).astype(np.uint8)

        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)
            depth = ops.resize(depth, height=h, width=w)

        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        mask = ops.resize(mask, height=self.input_hw[0], width=self.input_hw[1])
        depth = ops.resize(depth, height=self.input_hw[0], width=self.input_hw[1])
        assert all([x.dtype == np.uint8 for x in [image, mask, depth]])

        # Apply data augmentation
        transformed = self.trains(image=image, mask=mask, depth=depth)
        image = transformed["image"]
        mask = transformed["mask"]
        depth = transformed["depth"]

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).gt(0).float().unsqueeze(0)
        depth = torch.from_numpy(depth).div(255).float().unsqueeze(0)

        return dict(
            data={
                "image": image,
                "mask": mask,
                "depth": depth,
                "foreground_desc": foreground_desc,
                "background_desc": background_desc,
                "unique_id": unique_id
            },
            info={"text": class_name}
        )

    def __len__(self):
        return len(self.total_data_paths)


class ImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: list, descriptions_json_path: str):
        super().__init__()
        self.input_hw = input_hw
        self.descriptions_json_path = descriptions_json_path
        self.descriptions = self._load_descriptions()

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "test":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TE_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TE_MASK_DIR, unique_id + mask_suffix)

            # Check whether the description exists
            if unique_id not in self.descriptions:
                logger.warning(f"Description for {unique_id} was not found; using the default description")

            self.total_data_paths.append((class_name, image_path, mask_path, unique_id))

        logger.info(f"[TestSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

    def _load_descriptions(self):
        """Load the description JSON file."""
        if not os.path.exists(self.descriptions_json_path):
            logger.warning(f"Description file does not exist: {self.descriptions_json_path}")
            return {}

        try:
            with open(self.descriptions_json_path, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            logger.info(f"Loaded {len(descriptions)} descriptions")
            return descriptions
        except Exception as e:
            logger.error(f"Failed to load description file: {e}")
            return {}

    def __getitem__(self, index):
        class_name, image_path, mask_path, unique_id = self.total_data_paths[index]

        # Get the corresponding description
        desc = self.descriptions.get(unique_id, {})
        foreground_desc = desc.get("foreground_description", f"A camouflaged {class_name}")
        background_desc = desc.get("background_description", "The surrounding environment")

        image = io.read_color_array(image_path)
        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)

        return dict(
            data={
                "image": image,
                "foreground_desc": foreground_desc,
                "background_desc": background_desc,
                "unique_id": unique_id
            },
            info=dict(text=class_name, mask_path=mask_path, group_name="image")
        )

    def __len__(self):
        return len(self.total_data_paths)

@torch.no_grad()
def test(model, cfg, metric_names=("sm", "wfm", "mae", "fm", "em", "iou")):
    te_dataset = ImageTestDataset(dataset_info=cfg.root_info, input_hw=cfg.test.input_hw, descriptions_json_path=cfg.descriptions_json)
    te_loader = data.DataLoader(te_dataset, cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True)
    # model: trained model object
    # cfg: configuration object containing test settings
    # metric_names: evaluation metrics to compute
    if cfg.test.save_results:
        save_path = cfg.path.save
        logger.info(f"Results will be saved into {save_path}")
    else:
        save_path = ""

    model.eval()
    dataset_classes = te_loader.dataset.classes
    metricer = recorder.OVCOSMetricer(class_names=dataset_classes, metric_names=metric_names)
    for batch in tqdm(te_loader, total=len(te_loader), ncols=79, desc="[EVAL]"):
        batch_images = pt_utils.to_device(batch["data"], device=cfg.device)  # B,1,H,W
        gt_classes = batch["info"]["text"]
        outputs = model(data=batch_images, gt_classes=gt_classes, class_names=dataset_classes,
                       text_generator_model_path=cfg.text_generator_model_path,
                       test_vig_features_path=cfg.test_vig_features_path if hasattr(cfg, 'test_vig_features_path') else "")

        probs = outputs["prob"].squeeze(1).cpu().detach().numpy()  # B,H,W
        mask_paths = batch["info"]["mask_path"]
        for idx_in_batch, pred in enumerate(probs):
            mask_path = Path(mask_paths[idx_in_batch])
            mask = io.read_gray_array(mask_path.as_posix(), thr=0)
            mask = (mask * 255).astype(np.uint8)
            mask_h, mask_w = mask.shape

            pred = ops.minmax(pred)
            pred = ops.resize(pred, height=mask_h, width=mask_w)

            pre_cls = outputs["classes"][idx_in_batch]
            gt_cls = gt_classes[idx_in_batch]

            if save_path:
                # The predicted object is stored in brackets before the file name
                ops.save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_path)
            metricer.step(
                pre=(pred * 255).astype(np.uint8),
                gt=mask,
                pre_cls=pre_cls,
                gt_cls=gt_cls,
                gt_path=mask_path.as_posix(),
            )
    avg_ovcos_results = metricer.show()
    logger.info(str(avg_ovcos_results))


def train(model, cfg):
    # Initialize
    tr_dataset = ImageTrainDataset(dataset_info=cfg.root_info, input_hw=cfg.train.input_hw,
                                   descriptions_json_path=cfg.descriptions_json)
    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,  # Drop the last incomplete batch
        pin_memory=True,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )
    ## Training counter for epoch/iteration management
    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    # Learning rate scheduler
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    logger.info(
        f"Trainable Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if v.requires_grad))}"
    )
    logger.info(
        f"Fixed Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if not v.requires_grad))}"
    )

    scaler = pipeline.Scaler(optimizer=optimizer)
    logger.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()
    logger.info(f"Image Mean: {model.normalizer.mean.flatten()}, Image Std: {model.normalizer.std.flatten()}")

    # Main training loop
    train_start_time = time.perf_counter()
    for curr_epoch in range(counter.num_epochs):
        logger.info(f"Exp_Name: {cfg.exp_name}")

        model.train()
        # an epoch starts
        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)  # update learning rate

            # batch["data"] includes string fields; to_device only handles tensors or tensor containers

            data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
            gt_classes = batch["info"]["text"]
            outputs = model(
                data=data_batch,
                gt_classes=gt_classes,
                class_names=tr_dataset.classes,
                iter_percentage=counter.curr_percent,
                train_vig_features_path=cfg.train_vig_features_path if hasattr(cfg, 'train_vig_features_path') else ""
            )
## Compute gradients
            loss = outputs["loss"]
            loss_str = outputs["loss_str"]
            loss = loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            # Gradient update
            if counter.every_n_iters(cfg.train.grad_acc_step):  # Accumulates scaled gradients.
                scaler.update_grad()

            item_loss = loss.item()
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(value=item_loss, num=data_shape[0])

            # Log and visualize results

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem = f"{torch.cuda.max_memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                eta_seconds = iter_time_recorder.avg * (counter.num_total_iters - counter.curr_iter - 1)
                eta_string = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                progress = (
                    f"{counter.curr_iter}:{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )
                loss_info = f"{loss_str} (M:{loss_recorder.global_avg:.5f}/C:{item_loss:.5f})"
                lr_info = f"LR: {optimizer.lr_string()}"
                logger.info(f"{eta_string}({gpu_mem}) | {progress} | {lr_info} | {loss_info} | {data_shape}")

            if counter.curr_iter < 3:  # plot some batches of the training phase
                recorder.plot_results(
                    dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], **outputs["vis"]),
                    save_path=os.path.join(cfg.path.pth_log, "img", f"iter_{counter.curr_iter}.jpg"),
                )

            iter_time_recorder.update(value=time.perf_counter() - iter_start_time)
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        if curr_epoch < 3:
            recorder.plot_results(
                dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], **outputs["vis"]),
                save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{curr_epoch}.jpg"),
            )

        counter.update_epoch_counter()
        # an epoch ends

    io.save_weight(model=model, save_path=cfg.path.final_state_net, suffix="-final")

    total_train_time = time.perf_counter() - train_start_time
    total_other_time = datetime.timedelta(seconds=int(total_train_time - iter_time_recorder.global_sum))
    logger.info(f"Total Time: {datetime.timedelta(seconds=int(total_train_time))} ({total_other_time} on others)")


def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    # Removed the required flag
    parser.add_argument("--config", default="****", type=str)
    parser.add_argument("--root-info", default="****", type=str)
    # Change the model name here
    parser.add_argument("--model-name",default = 'OLCA', type=str)
    # Set the image-to-text model path here
    parser.add_argument("--text_generator_model_path", default='', type=str)
    # No longer loaded here
    # parser.add_argument("--text_generator_model_path", default="****", type=str)
    # If specified, load the pretrained model; keep this empty during training if needed
    # parser.add_argument("--load-from", type=str)  #  for training
    parser.add_argument("--load-from", default="****",type=str) # Load from a pretrained model for evaluation
    # Evaluation flag
    parser.add_argument("--evaluate", action="store_false")
    parser.add_argument("--info", type=str)
    parser.add_argument("--descriptions-json",
                        default="****",
                        type=str, help="Description JSON file path")
    # Previous description path
    # parser.add_argument("--descriptions-json",
    #                     default="****",
    #                     type=str, help="Description JSON file path")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))
    cfg.descriptions_json = args.descriptions_json  # Add the description file path to the config


    with open(cfg.root_info, mode="r") as f:
        cfg.root_info = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, "outputs")
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.device = "cuda:3"

    # Create all required directories in advance
    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    logger.add(cfg.path.log, level="INFO", format=logger_format)
    logger.info(cfg.pretty_text)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)

    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model_code = inspect.getsource(model_class)
    model = model_class(text_generator_model_path=cfg.text_generator_model_path,
                        use_prompts=getattr(cfg, 'use_visual_prompts', False),
                        prompt_config=getattr(cfg, 'prompt_config', None))
    logger.info(model_code)
    model.to(cfg.device)
    torch.set_float32_matmul_precision("high")

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=False)

    if not cfg.evaluate:
        train(model=model, cfg=cfg)
    else:
        cfg.test.save_results = True

    if cfg.evaluate or cfg.has_test:
        test(model=model, cfg=cfg)

    logger.info("End training...")


if __name__ == "__main__":
    main()
