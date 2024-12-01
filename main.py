from pickle import FALSE
from socket import IPPROTO_UDP
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from  matplotlib import pyplot as plt
import argparse
import yaml
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_mllm import get_reflected_text_from_img, get_mask, fuse_mask, DotDict, printd, mkdir
import os
from diffusers import StableDiffusionInpaintPipeline

## configs
if __name__ == '__main__':
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mydemo.yaml')
    parser.add_argument('--visualization', action='store_true')

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
data_args = config['test_dataset']
model_args = DotDict(config)
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

## get data
dataset = datasets.make(data_args['dataset'])
dataset = datasets.make(data_args['wrapper'], args={'dataset': dataset})
loader = DataLoader(dataset, batch_size=data_args['batch_size'],
                    num_workers=8)
paths_img = dataset.dataset.paths_img
data_len = len(paths_img)
printd(f"dataset size:\t {len(paths_img)}")

## save dir
config_name = args.config.split("/")[-1][:-5]
save_path_dir = f'output_img/{config_name}/'
mkdir(save_path_dir)

## load pretrained model
# CLIP surgery, SAM
from segment_anything import sam_model_registry, SamPredictor
from clip.clip_surgery_model import CLIPSurgery
import clip
sam = sam_model_registry[model_args.sam_model_type](checkpoint=model_args.sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
clip_params={ 'attn_qkv_strategy':model_args.clip_attn_qkv_strategy}
clip_model, _ = clip.load(model_args.clip_model, device=device, params=clip_params)
clip_model_ori, _ = clip.load(model_args.clip_model_ori, device=device, params=clip_params)
clip_model.eval()
clip_model_ori.eval()
# MLLM
llm_dict=None
if model_args.llm=='LLaVA':
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
    disable_torch_init()
    print(f'llava pretrained model: {model_args.model_path}')
    model_path = os.path.expanduser(model_args.model_path)
    model_args.model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_args.model_base,
        model_args.model_name
    )
    if 'llama-2' in model_args.model_name.lower(): # from clip.py
        conv_mode = "llava_llama_2"
    elif "v1" in model_args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    model_args.conv_mode = conv_mode
    llm_dict = {
        'model': model,
        'vis_processors':  image_processor,
        'tokenizer': tokenizer,
        'conv_mode': model_args.conv_mode,
        'temperature': model_args.temperature,
        'w_caption': model_args.LLaVA_w_caption,
    }
elif model_args.llm=='Mix':
    None
else:
    exit(f'unknow LLM: {model_args.llm}')


## metrics
import utils
metric_fn = utils.calc_cod
metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
val_metric1 = utils.Averager()
val_metric2 = utils.Averager()
val_metric3 = utils.Averager()
val_metric4 = utils.Averager()
text_similarity = []
miou_similarity = []
miou_after_similarity = []

## run model
printd('Start inference...')
for s_i, img_path, pairs in zip(range(data_len), paths_img, loader):
    printd(img_path)
    pil_img = Image.open(img_path).convert("RGB")
    mask_last = None
    print(f'-------- interation 0 --------')
    (text, text_bg, similarity_text, bbox, prediction) = get_reflected_text_from_img(
        pil_img,
        clip_model_ori,
        None,
        img_path,
        mask_last,
        None,
        sd_pipe,
        model_args.prompt_q,
        0,
        llm_dict,
        model_args.use_gene_prompt,
        model_args.clip_use_bg_text,
        model_args
    )

    (mask_l, mask_logit_origin_l, num_l, vis_dict, text_list, prediction_list) = get_mask(
        pil_img,
        text,
        bbox,
        sam_predictor,
        sd_pipe,
        clip_model,
        clip_model_ori,
        img_path,
        model_args,
        device,
        llm_dict=llm_dict,
        text_bg=text_bg,
        is_visualization=True
    )
    recursive_times = len(mask_l)
    predict_list = []
    predict_list = [prediction] + prediction_list
    
    vis_mask_acc, vis_mask_logit_acc = fuse_mask(
        mask_logit_origin_l,
        sam_predictor.model.mask_threshold,
        predict_list
    ) # fuse masks from different iterations

    ## get metric
    tensor_gt = pairs['gt']
    tensor_gt[tensor_gt != 0] = 1.0
    inp_size = 1024

    mask_transform = transforms.Compose([
                    transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
    
    # get metric of mask closest to fused mask
    mask_delta_l = [np.sum((mask_i - vis_mask_acc)**2) for mask_i in mask_l]  
    idxMaskSim = np.argmin(mask_delta_l)

    vis_tensor = Image.fromarray(mask_l[idxMaskSim].astype('uint8'))
    save_path_dir = f'output_img/CHAM_vis_m/'
    img_name = img_path.split('/')[-1][:-4]
    combined_img_name = os.path.join(save_path_dir, f'{img_name}_final.jpg')
    vis_tensor.save(combined_img_name)
    vis_tensor = mask_transform(vis_tensor)[0].view(1, 1, inp_size, inp_size)
    result1, result2, result3, result4 = metric_fn(vis_tensor, tensor_gt)
    val_metric1.add(result1, tensor_gt.shape[0])
    val_metric2.add(result2, tensor_gt.shape[0])
    val_metric3.add(result3, tensor_gt.shape[0])
    val_metric4.add(result4, tensor_gt.shape[0])

printd('End inference...')
print(f'\ncloset to fuse (formated):\n\
            {round(val_metric4.item(),4):.3f}\t\
            {round(val_metric3.item(),4):.3f}\t\
            {round(val_metric2.item(),4):.3f}\t\
            {round(val_metric1.item(),4):.3f}\t')
