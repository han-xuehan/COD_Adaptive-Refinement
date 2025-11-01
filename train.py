import matplotlib.pyplot as plt
import os
import numpy as np
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
import shutil
import glob
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple, Type
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer, AutoProcessor, MambaModel, BlipProcessor, \
    BlipForConditionalGeneration
from functools import partial
from utils_downstream.saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, \
    cal_acc
import re
import cv2
from scipy.ndimage import filters
from torch.optim.lr_scheduler import CosineAnnealingLR
from box import scoremap2bbox
import spacy
import matplotlib
import warnings
from toolbox import get_logger,averageMeter
from datetime import date
from datetime import datetime
warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")
torch.manual_seed(2024)
torch.cuda.empty_cache()

def eval_psnr(loader, model, mamba_model, cam_model, model_save_path,seg_loss,ce_loss, eval_type=None, device=None):
    model.eval()
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    test_loss_meter = averageMeter()
    test_loss_meter.reset()
    mae, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()

    for step, (image, gt2D, img_1024_ori, filename) in enumerate(loader):

        name = os.path.splitext(os.path.basename(filename[0]))[0]
        image, gt2D = image.to(device), gt2D.to(device)
        img_1024_ori = img_1024_ori.to(device)

        save_dir = os.path.join('BLIP_info', "CHAMELEON")
        load_path = os.path.join(save_dir, f"{name}.pt")
        loaded_data = torch.load(load_path)
        caption = loaded_data["caption"]
        merged_cam = loaded_data["merged_cam"]
        image_features = loaded_data["image_features"]
        img_emb_textretrieval = loaded_data["img_emb_textretrieval"]
        vision_attention = loaded_data["vision_attention"]

        text_input= cam_model(caption)
        img = img_1024_ori.squeeze(0).cpu()
        rgb_image = np.float32(img) / 255
        gradcam_image, attMap_without_overlap = getAttMap(rgb_image, merged_cam.cpu().numpy())

        with torch.no_grad():
            pred, adapter, trans_mat, attention_adapter = model(image.float(), img_1024_ori, mamba_model, gradcam_image,
                                                                attMap_without_overlap, merged_cam, vision_attention,
                                                                text_input, image_features, img_emb_textretrieval)
            loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
            test_loss_meter.update(loss.item())
            pred = torch.sigmoid(pred)

            res = pred.squeeze().squeeze().cpu().numpy()
            gt = gt2D.squeeze().squeeze().cpu().numpy()

            mae.update(res, gt)
            sm.update(res, gt)

            em.update(res, gt)
            wfm.update(res, gt)
            m_dice.update(res, gt)
            m_iou.update(res, gt)
            ber.update(res, gt)

        if pbar is not None:
            pbar.update(1)

    MAE = mae.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()
    test_loss = test_loss_meter.avg

    if pbar is not None:
        pbar.close()

    return sm, em, wfm, MAE, test_loss


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "GT/")
        self.img_path = join(data_root, "Imgs/")
        self.gt_path_files = sorted([self.gt_path + f for f in os.listdir(self.gt_path) if f.endswith('.png')])
        self.img_path_files = sorted([self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg')])
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_1024_ori = Image.open(self.img_path_files[index]).convert('RGB')

        gt = Image.open(self.gt_path_files[index]).convert('L')  # multiple labels [0, 1,4,5...], (256,256)

        img_1024 = self.img_transform(img_1024_ori)
        gt = self.mask_transform(gt)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            np.array(img_1024_ori),
            self.img_path_files[index]
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/TrainDataset",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="COD_CAM")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_h_4b8939.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=40)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0002, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name)
device = torch.device(args.device)

# %% set up model
class VLSAM(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.no_mask_embed = nn.Embedding(1, 256)

    def forward(self, image, img_1024_ori, mamba_model, gradcam_image, attMap_without_overlap,
                merged_cam, vision_attention, text_input,
                image_features, img_emb_textretrieval):
        # ——————————————————————————————————————————————————————————————————————————
        image_embedding, adapter = self.image_encoder(image, img_emb_textretrieval)
        # ——————————————————————————————————————————————————————————————————————————
        refine_gradcam_image, refine_attMap_without_overlap, adapter, trans_mat, attention_adapter = self.refinement(
            attMap_without_overlap, merged_cam, img_1024_ori, vision_attention, adapter, threshold=0.85)
        binary_map = np.where(refine_attMap_without_overlap > 0.85, 255, 0).astype(np.uint8)
        binary = cv2.resize(binary_map, (256, 256))
        binary_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        binary = binary_transform(binary)
        # ——————————————————————————————————————————————————————————————————————————
        mamba_outputs = mamba_model(**text_input)
        text_features = mamba_outputs.last_hidden_state
        # ——————————————————————————————————————————————————————————————————————————
        mamba_text = text_features.view(1, -1, 256)
        blip_img = image_features.view(1, -1, 256)
        sparse_embeddings = torch.cat((mamba_text, blip_img), dim=1)
        # ——————————————————————— prompt ———————————————————————————————————————

        prompt = torch.tensor(binary).to(device)
        _, dense_embeddings = self.prompt_encoder(masks=prompt, points=None, boxes=None)

        bs, c, h, w = image_embedding.shape
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.pe_layer((64, 64)).unsqueeze(0),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks, adapter, trans_mat, attention_adapter

    def refinement(self, attMap_without_overlap, merged_cam, img_1024_ori, vision_attention, adapter, threshold=0.85):
        box, cnt = scoremap2bbox(scoremap=attMap_without_overlap, threshold=threshold, multi_contour_eval=True)
        aff_mask = torch.zeros((attMap_without_overlap.shape[0], attMap_without_overlap.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = np.asarray(box)[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 255
        mask = cv2.resize(aff_mask.detach().numpy(), (24, 24))
        mean_attention = vision_attention.mean(dim=1)
        attention_weight = mean_attention[:, 1:, 1:][-1]
        attn_weight = attention_weight.float()
        trans_mat = attn_weight / torch.sum(attn_weight, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
        trans_mat = torch.matmul(trans_mat, trans_mat)
        # ——————————————————adapter————————————————————————————————————————
        adapter_upsampled = F.interpolate(
            adapter.permute(0, 3, 1, 2),  # [1, 1280, 64, 64] → [1, 1280, 576, 576]
            size=(576, 576),
            mode='bilinear',
            align_corners=False
        )
        adapter = torch.sigmoid(adapter_upsampled.mean(dim=1))
        attention_adapter = torch.tensor(trans_mat) * adapter.squeeze()
        # ——————————————————adapter————————————————————————————————————————
        mask = torch.tensor(mask).reshape(1, merged_cam.shape[0] * merged_cam.shape[1])
        att_adapter = attention_adapter.detach().cpu().numpy() * mask.detach().cpu().numpy()
        cam_to_refine = torch.FloatTensor(merged_cam)
        cam_to_refine = cam_to_refine.view(-1, 1)
        cam_refined = torch.matmul(torch.tensor(att_adapter), cam_to_refine).reshape(24, 24)
        img = img_1024_ori.squeeze(0).cpu()
        rgb_image = np.float32(img) / 255
        refine_gradcam_image, refine_attMap_without_overlap = getAttMap(rgb_image, cam_refined.detach().numpy())
        return refine_gradcam_image, refine_attMap_without_overlap, adapter, trans_mat, attention_adapter


class BLIP_ITM(nn.Module):
    def __init__(self,
                 tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, caption):
        device = torch.device("cuda:0")
        text_input = self.tokenizer(caption, return_tensors="pt").to(device)
        return text_input

def getAttMap(img, attMap, blur=True, overlap=True):  # 获取attention Map与原图重叠的CAM图像
    if isinstance(attMap, torch.Tensor):
        attMap = attMap.cpu().numpy()  # 转换为 NumPy 数组
    attMap -= attMap.min()
    if attMap.max() > 0:  # 将 attMap 的值归一化到 [0, 1] 区间：归一化的目的是为了帮助后续映射注意力值到热力图的颜色上时颜色之间的差距较大更好区分
        attMap /= attMap.max()

    # 将注意力图的尺寸调整为与原始图像 img 相同的大小。(在示例图当中是（480, 640)本来image.shape是(480,640,3)
    attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    attMap_without_overlap = attMap
    # print("???????????????",img.shape)
    if blur:  # smooth一下
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)  # 删除 Alpha 通道
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
            attMap.shape + (1,)) * attMapV
        # attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**1).reshape(attMap.shape+(1,)) * attMapV
        # 0.7:这里的数字越大热力图越淡如果等于0那么只有热力图没用原图的叠加
        # 1*(1-attMap**0.7).reshape(attMap.shape + (1,))：attMap**0.7由于attMap的范围在[0,1]所以小数次幂会让对应的数字更大
    return attMap, attMap_without_overlap


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    current_date = date.today()
    logdir = os.path.join(model_save_path, f'log/{current_date}')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    vlsam_model = VLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    for name, param in vlsam_model.image_encoder.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    vlsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in vlsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in vlsam_model.parameters() if p.requires_grad),
    )

    adapter_params = vlsam_model.image_encoder.adapter.parameters()

    img_mask_encdec_params = list(
        vlsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": args.lr * 0.1},
            {"params": img_mask_encdec_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay
    )

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Group {i}, Learning Rate = {param_group['lr']}")

    lr_scheduler = CosineAnnealingLR(optimizer, 20, eta_min=1.0e-6)
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    testloss=[]
    best_loss = 1e10
    best_accuracy = 0
    train_dataset = NpyDataset(args.tr_npy_path)

    test_dataset = NpyDataset('data/TestDataset/CHAMELEON')
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            vlsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    tokenizer = AutoTokenizer.from_pretrained("./mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("./mamba-130m-hf").to(device)

    csv_path = os.path.join(model_save_path, 'train.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sm', 'Em', 'wFm', 'Mae'])
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            for step, (image, gt2D, img_1024_ori, filename) in enumerate(tqdm(train_dataloader)):

                name = os.path.splitext(os.path.basename(filename[0]))[0]
                optimizer.zero_grad()
                image, gt2D = image.to(device), gt2D.to(device)
                img_1024_ori = img_1024_ori.to(device)

                save_dir = os.path.join('BLIP_info', "info_save")
                load_path = os.path.join(save_dir, f"{name}_train.pt")
                loaded_data = torch.load(load_path)
                caption = loaded_data["caption"]
                merged_cam = loaded_data["merged_cam"]
                image_features=loaded_data["image_features"]
                img_emb_textretrieval=loaded_data["img_emb_textretrieval"]
                vision_attention=loaded_data["vision_attention"]


                cam_model = BLIP_ITM(tokenizer)
                text_input = cam_model(caption)


                img = img_1024_ori.squeeze(0).cpu()
                rgb_image = np.float32(img) / 255
                gradcam_image, attMap_without_overlap = getAttMap(rgb_image, merged_cam.cpu().numpy())

                if args.use_amp:
                    ## AMP
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        medsam_pred, adapter, trans_mat, attention_adapter = vlsam_model(image)
                        loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                            medsam_pred, gt2D.float()
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    medsam_pred, adapter, trans_mat, attention_adapter = vlsam_model(image.float(), img_1024_ori,
                                                                                     mamba_model, gradcam_image,
                                                                                     attMap_without_overlap, merged_cam,
                                                                                     vision_attention, text_input,
                                                                                     image_features,
                                                                                     img_emb_textretrieval)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                iter_num += 1
                lr_scheduler.step()

            result1, result2, result3, result4, test_loss = eval_psnr(test_dataloader, vlsam_model, mamba_model,cam_model, model_save_path,seg_loss,ce_loss, eval_type='cod',
                                                           device=device)
            testloss.append(test_loss)
            logger.info(
                f'Iter | [{epoch + 1:3d}/40] Sm:{result1:.5f} Em:{result2:.5f} wFm:{result3:.5f} Mae:{result4:.5f}')
            writer.writerow([result1, result2, result3, result4])

            epoch_loss /= step
            epoch_accuracy = (result1 + result2 + result3) / 3
            losses.append(epoch_loss)
            if args.use_wandb:
                wandb.log({"epoch_loss": epoch_loss})
            print(
                f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
            )
            ## save the latest model
            checkpoint = {
                "model": vlsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "vlsam_model_latest.pth"))
            ## save the best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                checkpoint = {
                    "model": vlsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, join(model_save_path, "vlsam_model_best.pth"))

            # %% plot loss
            plt.plot(losses, label="Train Loss")
            plt.plot(testloss, label="Test Loss", linestyle="--")
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(join(model_save_path, args.task_name + "_loss.png"))
            plt.close()


if __name__ == "__main__":
    main()
