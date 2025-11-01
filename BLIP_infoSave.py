import numpy as np
import os
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import re
from models.modeling_blip import BlipForImageTextRetrieval
import spacy
import warnings
warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")
torch.manual_seed(2024)
torch.cuda.empty_cache()

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/TrainDataset",
    help="path to training npy files; two subfolders: gts and imgs",
)

parser.add_argument("-num_epochs", type=int, default=1)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

BLIP_info_save_path = 'BLIP_info'
device = torch.device(args.device)

def eval_psnr(loader, cam_model, block_num, BLIP_info_save_path,TestData_Name,device):
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    for step, (image, gt2D, img_1024_ori, filename) in enumerate(tqdm(loader)):
        name = os.path.splitext(os.path.basename(filename[0]))[0]
        img_1024_ori = img_1024_ori.to(device)
        output, caption, vision_attention, image_features, img_emb_textretrieval = cam_model(
            img_1024_ori)
        loss_cam = output[:, 1].sum()
        cam_model.zero_grad()
        loss_cam.backward()
        with torch.no_grad():
            grads = cam_model.model_textretrieval.text_encoder.encoder.layer[
                block_num].crossattention.self.get_attn_gradients()
            cams = cam_model.model_textretrieval.text_encoder.encoder.layer[
                block_num].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(1, 12, -1, 24, 24)  # * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(1, 12, -1, 24, 24)  # * mask

            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()
        merged_cam = torch.mean(gradcam, dim=0)

        save_test = {
            "merged_cam": merged_cam,
            "vision_attention": vision_attention[-1],
            "caption": caption,
            "image_features": image_features,
            "img_emb_textretrieval": img_emb_textretrieval,
        }
        save_dir = os.path.join(BLIP_info_save_path, TestData_Name)
        os.makedirs(save_dir, exist_ok=True)
        save_test_path = os.path.join(save_dir, f"{name}.pt")
        torch.save(save_test, save_test_path)

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
        gt = Image.open(self.gt_path_files[index]).convert('L')

        img_1024 = self.img_transform(img_1024_ori)
        gt = self.mask_transform(gt)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            np.array(img_1024_ori),
            self.img_path_files[index]
        )

class BLIP_ITM(nn.Module):
    def __init__(self,
                 processor, vlm_model_generate, model_textretrieval):
        super().__init__()
        self.processor = processor
        self.vlm_model_generate = vlm_model_generate
        self.model_textretrieval = model_textretrieval

    def pre_caption(self, caption, max_words=30):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '', 
            caption.lower(), 
        ).replace('-', ' ').replace('/', ' ')

        caption = re.sub(
            r"\s{2,}", 
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        return caption

    def forward(self, image):
        device = torch.device("cuda:0")
        vlm_inputs = self.processor(image, return_tensors="pt").to(device)
        vlm_outputs = self.vlm_model_generate.generate(**vlm_inputs, output_hidden_states=True)
        vision_outputs = self.vlm_model_generate.vision_model(**vlm_inputs)
        image_features = vision_outputs.last_hidden_state[:, 1:, :]

        text = self.processor.decode(vlm_outputs[0], skip_special_tokens=True)
        caption = self.pre_caption(text)
        doc = nlp(caption)

        # extract first noun
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        text = nouns[0]

        textretrieval_inputs = self.processor(images=image, text=text, return_tensors="pt").to(device)
        textretrieval_outputs, vision_attention, vision_textretrieval = self.model_textretrieval(**textretrieval_inputs)
        img_emb_textretrieval = vision_textretrieval.last_hidden_state[:, 1:, :]
        logits_per_image = textretrieval_outputs.itm_score

        return logits_per_image, text, vision_attention, image_features, img_emb_textretrieval

def main():
    os.makedirs(BLIP_info_save_path, exist_ok=True)
    num_epochs = args.num_epochs
    train_dataset = NpyDataset(args.tr_npy_path)
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    start_epoch = 0
    processor = BlipProcessor.from_pretrained("./blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained("./blip-image-captioning-large").to(device)
    model_textretrieval = BlipForImageTextRetrieval.from_pretrained("./blip-itm-base-coco").to(device)
    
    for epoch in range(start_epoch, num_epochs):
        for step, (image, gt2D, img_1024_ori, filename) in enumerate(tqdm(train_dataloader)):
            name = os.path.splitext(os.path.basename(filename[0]))[0]
            img_1024_ori = img_1024_ori.to(device)
            block_num = 8
            cam_model = BLIP_ITM(processor, vlm_model, model_textretrieval)
            cam_model.eval()
            cam_model.model_textretrieval.text_encoder.encoder.layer[
                block_num].crossattention.self.save_attention = True
            output, caption, vision_attention, image_features, img_emb_textretrieval = cam_model(
                img_1024_ori)
            loss_cam = output[:, 1].sum()
            cam_model.zero_grad()
            loss_cam.backward()
            with torch.no_grad():
                grads = cam_model.model_textretrieval.text_encoder.encoder.layer[
                    block_num].crossattention.self.get_attn_gradients()
                cams = cam_model.model_textretrieval.text_encoder.encoder.layer[
                    block_num].crossattention.self.get_attention_map()

                cams = cams[:, :, :, 1:].reshape(1, 12, -1, 24, 24)  # * mask
                grads = grads[:, :, :, 1:].clamp(0).reshape(1, 12, -1, 24, 24)  # * mask

                gradcam = cams * grads
                gradcam = gradcam[0].mean(0).cpu().detach()
            merged_cam = torch.mean(gradcam, dim=0)

            save_train = {
                    "merged_cam": merged_cam,
                    "vision_attention": vision_attention[-1],
                    "caption":caption,
                    "image_features": image_features,
                    "img_emb_textretrieval": img_emb_textretrieval,
            }

            save_dir = os.path.join(BLIP_info_save_path, "info_save")
            os.makedirs(save_dir, exist_ok=True)
            save_train_path = os.path.join(save_dir, f"{name}_train.pt")
            torch.save(save_train, save_train_path)

        print("save BLIP info for train data end")
#---------------------------------------------------------------------------------------------------------------------
        print("save BLIP info for test data start")

        TestData_Name = ["CHAMELEON", "CAMO", "COD10K"]
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
            eval_psnr(test_dataloader, cam_model,
                          block_num, BLIP_info_save_path, Testdata, device=device)

        print("save end")

if __name__ == "__main__":
    main()
