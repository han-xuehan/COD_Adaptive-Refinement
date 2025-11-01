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


def eval_psnr(loader, model, mamba_model, cam_model, Testdata,eval_type=None, device=None):
    model.eval()
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    test_loss_meter = averageMeter()
    test_loss_meter.reset()
    mae, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()

    for step, (image, gt2D, img_1024_ori, filename) in enumerate(loader):

        name = os.path.splitext(os.path.basename(filename[0]))[0]
        image, gt2D = image.to(device), gt2D.to(device)
        img_1024_ori = img_1024_ori.to(device)

        save_dir = os.path.join('BLIP_info', Testdata)
        load_path = os.path.join(save_dir, f"{name}.pt")
        loaded_data = torch.load(load_path)
        caption = loaded_data["caption"]
        merged_cam = loaded_data["merged_cam"]
        image_features = loaded_data["image_features"]
        img_emb_textretrieval = loaded_data["img_emb_textretrieval"]
        vision_attention = loaded_data["vision_attention"]

        text_input = cam_model(caption)
        img = img_1024_ori.squeeze(0).cpu()
        rgb_image = np.float32(img) / 255
        gradcam_image, attMap_without_overlap = getAttMap(rgb_image, merged_cam.cpu().numpy())

        with torch.no_grad():
            pred, adapter, trans_mat, attention_adapter = model(image.float(), img_1024_ori, mamba_model, gradcam_image,
                                                                attMap_without_overlap, merged_cam, vision_attention,
                                                                text_input, image_features, img_emb_textretrieval)
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
    if pbar is not None:
        pbar.close()

    return sm, em, wfm, MAE

    
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
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument("-logdir", type=str, default="work_dir/iccv2025/")
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_h_4b8939.pth"
)

parser.add_argument("-work_dir", type=str, default="work_dir")

parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

device = torch.device(args.device)
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
        # trans_mat = trans_mat.detach().cpu().numpy()  * mask.detach().cpu().numpy()
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
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    vlsam_model = VLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    checkpoint_path = os.path.join(args.logdir, "vlsam_model_best.pth")
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model']
    vlsam_model.load_state_dict(model_state_dict)

    tokenizer = AutoTokenizer.from_pretrained("./mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("./mamba-130m-hf").to(device)

    cam_model = BLIP_ITM(tokenizer)
    TestData_Name=["CHAMELEON","CAMO","COD10K"]

    csv_path = os.path.join(args.logdir, 'test.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['TestData','Sm', 'Em', 'wFm', 'Mae'])

        for Testdata in TestData_Name:
            dataset_path = f"data/TestDataset/{Testdata}"
            test_dataset = NpyDataset(dataset_path)

            print("Number of testing samples: ", len(test_dataset))
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            result1, result2, result3, result4 = eval_psnr(test_dataloader, vlsam_model, mamba_model, cam_model,
                                                           Testdata,eval_type='cod',
                                                           device=device)

            print({'TestData':Testdata})
            print({'Sm': result1})
            print({'Em': result2})
            print({'wFm': result3})
            print({'Mae': result4})
            writer.writerow([Testdata, result1, result2, result3, result4])

if __name__ == "__main__":
    main()
