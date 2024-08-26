import clip
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
import datetime
import os
from utils import show

BICUBIC = InterpolationMode.BICUBIC
eps = 1e-7

def fuse_mask(mask_logit_origin_l, sam_thr, possibility_list, fuse='avg'):
    num_mask = len(mask_logit_origin_l)
    if fuse=='avg':
        mask_logit_origin = sum(mask_logit_origin_l)/num_mask  #
    elif fuse == 'weight':
        total_weight = sum(possibility_list)
        normalized_weights = [w / total_weight for w in possibility_list]
        mask_logit_origin = np.zeros_like(mask_logit_origin_l[0])
        for i, mask_logit in enumerate(mask_logit_origin_l):
            mask_logit_origin = mask_logit_origin.astype('float64') 
            mask_logit_origin += normalized_weights[i] * mask_logit

    mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
    mask = mask_logit_origin > sam_thr
    mask = mask.astype('uint8')
    mask_logit *= 255
    mask_logit = mask_logit.astype('uint8')

    return mask, mask_logit


def get_mask(pil_img, text, bbox, sam_predictor, sd_pipe, clip_model, clip_model_ori, img_path, args, device='cuda', llm_dict=None, text_bg=None, is_visualization=False):
    text_list_individual = []
    num_l = []
    mask_l = []
    mask_logit_origin_l = []
    mask_logit_l = []
    vis_mask_logit_l = []
    bbox_list = []  # get the box prompt
    possibility_list = []
    vis_dict = {}
    text_list_individual.append(text[0])

    ori_image = np.array(pil_img)
    bbox_list.append(bbox)

    cur_image = ori_image
    with torch.no_grad():
        for i in range(args.recursive+1):
            
            if i>=1 and args.update_text:
                cur_image = cur_image.astype('uint8')
                if args.check_exist_each_iter and text==[]:
                    return None, mask_logit_origin_l, None, None, None, num_l, vis_dict
            print("instance-specific text prompt", text)
            masks_list, patch_img_list, patch_list, masks_weight_list, sm_list = [], [], [], [], []
            patches_list = args.patch_list
            for patch_num in patches_list:
                masks_list_patch_1, patch_img_1, patch_1, mask_weight_patch_1, sm_list_1  = Seg_custom(cur_image, text, bbox_list, clip_model, sam_predictor, i, args, device, patch_num, text_bg=text_bg, is_visualization=is_visualization)
                masks_list.extend(masks_list_patch_1)
                patch_img_list.extend(patch_img_1)
                patch_list.extend(patch_1)
                masks_weight_list.extend(mask_weight_patch_1)
                sm_list.extend(sm_list_1)

            np_img_combine, normalized_weighted_mask, _, _ = clip_similarity(patch_img_list, masks_list, sm_list, masks_weight_list, text[0], clip_model_ori)
            
            target_height, target_width = ori_image.shape[:2]
            mask_combine = cv2.resize(np_img_combine.squeeze(), (target_width, target_height), interpolation=cv2.INTER_CUBIC)                        
            mask_weight_all = cv2.resize(sm_list[0], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            mask_weight_all = np.repeat(mask_weight_all[:, :, np.newaxis], 3, axis=2)
            sm = cv2.resize(normalized_weighted_mask, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            show(Image.fromarray((mask_combine * 255).astype(np.uint8)), f"Output Mask (iter {i})" )
            sm1 = np.repeat(sm[:, :, np.newaxis], 3, axis=2)
            mask_image = Image.fromarray((normalized_weighted_mask * 255).astype(np.uint8))
            mask_array = np.array(mask_image)

            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles_img = np.zeros_like(mask_array)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)  
                cv2.rectangle(rectangles_img, (x, y), (x+w, y+h), (255), thickness=-1)  

            if args.clipInputEMA:
                cur_image = ori_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
            else:
                cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
            if i<args.recursive and args.update_text:
                print(f'-------- interation {i+1} --------')
                text, _, _, bbox, possibility = get_reflected_text_from_img(Image.fromarray(np.uint8(ori_image)), clip_model_ori, bbox, img_path, mask_image.convert('RGB'),1 - mask_weight_all, sd_pipe, args.prompt_q, i+1, llm_dict,
                                       args.use_gene_prompt, args.clip_use_bg_text, args)
                bbox_list.append(bbox)
                possibility_list.append(possibility)
                text_list_individual.append(text[0])

            vis_mask_logit_l.append((sm1 * 255).astype('uint8'))
            mask_logit_l.append(mask_combine)
            num_l.append(10)
            mask_l.append(mask_combine.squeeze())
            mask_logit_origin_l.append(sm)

            vis_dict = {
                        'sm_fg_bg_l': bbox_list,
            }

        return mask_l, mask_logit_origin_l, num_l, vis_dict, text_list_individual, possibility_list


def clip_surgery(np_img, text, model, args, device='cuda', text_bg=None, is_visualization=False):
    if is_visualization:
        sm_sub_l, sm_bg_sub_l = [], []
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    h, w = np_img.shape[:2]
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)
    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])
    # Extract redundant features from an empty string
    redundant_features = clip.encode_text_with_prompt_ensemble(model, [args.rdd_str], device)  # torch.Size([1, 512])
    # Prompt ensemble for text features with normalization
    text_features = clip.encode_text_with_prompt_ensemble(model, text, device)  # torch.Size([x, 512])
    if args.clip_use_bg_text:
        text_bg_features = clip.encode_text_with_prompt_ensemble(model, text_bg, device)  # torch.Size([x, 512])

    def _norm_sm(_sm, h, w):
        side = int(_sm.shape[0] ** 0.5)
        _sm = _sm.reshape(1, 1, side, side)
        _sm = torch.nn.functional.interpolate(_sm, (h, w), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
        _sm = (_sm - _sm.min()) / (_sm.max() - _sm.min())
        _sm = _sm.detach().cpu().numpy()
        return _sm
    # Combine features after removing redundant features and min-max norm
    sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
    sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
    sm_mean = sm_norm.mean(-1, keepdim=True)
    if is_visualization:
        sm_sub_l = [_norm_sm(sm_norm[..., i:i+1], h, w) for i in range( sm_norm.size()[-1] )]
        sm_mean_fg = _norm_sm(sm_mean, h, w)

    sm_mean_bg, sm_mean_fg_bg=None, None
    if args.clip_use_bg_text:
        sm_bg = clip.clip_feature_surgery(image_features, text_bg_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
        sm_norm_bg = (sm_bg - sm_bg.min(0, keepdim=True)[0]) / (sm_bg.max(0, keepdim=True)[0] - sm_bg.min(0, keepdim=True)[0])
        sm_mean_bg = sm_norm_bg.mean(-1, keepdim=True)
        if is_visualization:  sm_bg_sub_l = [_norm_sm(sm_norm_bg[...,i:i+1], h, w) for i in range(sm_norm_bg.size()[-1])]

        if args.clip_bg_strategy=='FgBgHm':
            sm_mean_fg_bg = sm_mean - sm_mean_bg
        else: # FgBgHmClamp
            sm_mean_fg_bg = torch.clamp(sm_mean - sm_mean_bg, 0, 1)

        sm_mean_fg_bg = (sm_mean_fg_bg - sm_mean_fg_bg.min(0, keepdim=True)[0]) / (sm_mean_fg_bg.max(0, keepdim=True)[0] - sm_mean_fg_bg.min(0, keepdim=True)[0])
        sm_mean_fg_bg_origin = sm_mean_fg_bg
        sm_mean = sm_mean_fg_bg_origin

    # expand similarity map to original image size, normalize. to apply to image for next iter

    sm1 = sm_mean
    sm_logit = _norm_sm(sm1, h, w)
    sm_mean_fg_bg = _norm_sm(sm_mean_fg_bg, h, w)
    if is_visualization and args.clip_use_bg_text:
        sm_mean_bg = _norm_sm(sm_mean_bg, h, w)
    clip_vis_dict={'sm_fg_bg':	sm_mean_fg_bg,}
    if is_visualization:
        clip_vis_dict={
            'sm_fg':	sm_mean_fg,
            'sm_bg':	sm_mean_bg,
            'sm_fg_bg':	sm_mean_fg_bg,
            'sm_sub_l':	sm_sub_l,
            'sm_bg_sub_l':	sm_bg_sub_l,}

    return sm, sm_mean, sm_logit, clip_vis_dict

template_q='Name of the {} in one word.'
template_bg_q='Name of the environment of the {} in one word.'
prompt_qkeys_dict={
    'TheCamo':          ['camouflaged animal'],
    'ThePolyp':         ['polyp'],
    'TheSkin':          ['Skin Lesion'],
}
prompt_q_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_q_dict.get(k) is None:
        prompt_q_dict[k] = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_dict[k]]
prompt_gene_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_gene_dict.get(k) is None:
        prompt_gene_dict[k] = [prompt_qkeys_dict[k], ['environment']]

def heatmap2points(sm_mean, np_img, args, attn_thr=-1):
    cv2_img = cv2.cvtColor(np_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    if attn_thr < 0:
        attn_thr = args.attn_thr
    map_l=[]
    p, l, map, _ = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=attn_thr,
                                                    down_sample=args.down_sample) # p: [pos (min->max), neg(max->min)]
    map_l.append(map)
    num = len(p) // 2
    points = p[num:] # negatives in the second half
    labels = [l[num:]]

    points = points + p[:num] # positive in first half
    labels.append(l[:num])
    labels = np.concatenate(labels, 0)
    return points, labels

#### utility ####
class DotDict:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)
def printd(str):
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt+'\t '+str)
def get_edge_img_path(mask_path, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return get_edge_img(binary_mask, img)
def get_edge_img(binary_mask, img):
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    edges = cv2.Canny(binary_mask, threshold1=30, threshold2=100)
    thicker_edges = cv2.dilate(edges, kernel, iterations=1)
    coord=(thicker_edges==255)
    img[...,:][coord]=np.array([255, 200,200])
    coord_fg = (binary_mask==255)
    coord_bg = (binary_mask==0)
    r = 0.2
    img[...,0][coord_fg] = img[...,0][coord_fg] * (1-r) + 255 * r
    img[...,2][coord_bg] = img[...,2][coord_bg] * (1-r) + 255 * r
    img = np.clip(img,0,255) 
    return img

def Seg_custom(cur_image, text, bbox_list, clip_model, sam_predictor, iter, args, device='cuda', patches=1, text_bg=None, is_visualization=None):
    cur_image = cur_image.astype(np.uint8)
    image_height, image_width = cur_image.shape[:2]
    blocks = [(0, 0, image_width, image_height)]

    # split image into various patches
    if patches == 0.5:
        center_left = image_width // 4
        center_upper = image_height // 4
        center_right = center_left + (image_width // 2)
        center_lower = center_upper + (image_height // 2)
        blocks = [(center_left, center_upper, center_right, center_lower)]
    if patches == 2:
        mid_width = image_width // 2
        blocks = ([(0, 0, mid_width, image_height), (mid_width, 0, image_width, image_height)])
        mid_height = image_height // 2
        blocks.extend([(0, 0, image_width, mid_height), (0, mid_height, image_width, image_height)])
    else:
        num_cuts = int(np.ceil(np.log2(patches))) 
        for _ in range(num_cuts):
            new_blocks = []
            for left, upper, right, lower in blocks:
                if (right - left) >= (lower - upper):
                    mid = (left + right) // 2
                    new_blocks.append((left, upper, mid, lower))
                    new_blocks.append((mid, upper, right, lower))
                else:
                    mid = (upper + lower) // 2
                    new_blocks.append((left, upper, right, mid))
                    new_blocks.append((left, mid, right, lower))
            blocks = new_blocks

    mask_weight = []
    sm_list = []
    mask_list, patch_match_list, patch_list = [], [], []
    for block in blocks:
        black_background = Image.new('L', (image_width, image_height), 0)
        black_background_ori = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        left, upper, right, lower = block
        patch = cur_image[upper:lower, left:right]             
        sm, sm_mean, sm_logit, clip_vis_dict = clip_surgery(patch,
                                                                text,
                                                                clip_model,
                                                                args, device='cuda',
                                                                text_bg=text_bg,
                                                                is_visualization=is_visualization)
        points, labels  = heatmap2points(sm_mean, patch, args)
        sam_predictor.set_image(patch)
        # Inference SAM with points from CLIP Surgery
        if args.post_mode =='MaxIOUBoxSAMInput':
            bbox_now = adjust_bbox_to_patch(bbox_list[iter], upper, lower, left, right)
            if len(points) == 0:
                    mask_logit_origin, scores, logits = sam_predictor.predict(box=bbox_now[None, :], multimask_output=True, return_logits=True)
            else:
                if len(bbox_now) != 0:
                    mask_logit_origin, scores, logits = sam_predictor.predict(box=bbox_now[None, :], point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True)
                else:
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True)
            mask_logit_origin = mask_logit_origin[np.argmax(scores)]
            # mask_logit_origin_blur = mask_logit_origin_blur[np.argmax(scores_blur)]
            mask = mask_logit_origin > sam_predictor.model.mask_threshold
            mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()

                
            if len(cur_image.shape) == 3:
                mask1 = mask[:, :, np.newaxis]

            masked_image = np.where(mask1 == 1, patch, 0)

        patch = Image.fromarray(masked_image)
        black_background_ori.paste(patch, (left, upper))
        patch_match_list.append(black_background_ori)
        patch_list.append(patch)
        mask_patch = Image.fromarray(mask)
        black_background.paste(mask_patch, (left, upper))
        mask_list.append(np.array(black_background))
        black_background_np = np.zeros((image_height, image_width), dtype=mask_logit.dtype)
        black_background_np[upper:upper+mask_logit.shape[0], left:left+mask_logit.shape[1]] = mask_logit
        mask_weight.append(black_background_np)
        black_background_np = np.zeros((image_height, image_width), dtype=mask_logit.dtype)
        sm_logit_squeezed = sm_logit.squeeze()
        black_background_np[upper:upper+mask_logit.shape[0], left:left+mask_logit.shape[1]] = sm_logit_squeezed
        sm_list.append(black_background_np)

    return mask_list, patch_match_list, patch_list, mask_weight, sm_list


def adjust_bbox_to_patch(bbox, upper, lower, left, right):
    if bbox[2] <= left or bbox[0] >= right or bbox[3] <= upper or bbox[1] >= lower:
        return []
    new_x_min = max(bbox[0], left) - left
    new_y_min = max(bbox[1], upper) - upper
    new_x_max = min(bbox[2], right) - left
    new_y_max = min(bbox[3], lower) - upper
    
    return np.array([new_x_min, new_y_min, new_x_max, new_y_max])


def get_reflected_text_from_img(pil_img, clip_model, bbox_last_iter, img_path, mask_last, sm, sd_pipe, prompt_q, iter, llm_dict, use_gene_prompt, get_bg_text, args,
                        reset_prompt_qkeys=False, new_prompt_qkeys_l=None,
                        bg_cat_list=[],
                        post_process_per_cat_fg=False):
    if use_gene_prompt:
        return prompt_gene_dict[args.prompt_q]
    else:  # use LLM model: LLaVA
        model = llm_dict['model']
        vis_processors = llm_dict['vis_processors']
        use_gene_prompt_fg=args.use_gene_prompt_fg
        if args.llm=='LLaVA' or args.llm=='LLaVA1.5':
            tokenizer = llm_dict['tokenizer']
            conv_mode = llm_dict['conv_mode']
            temperature = llm_dict['temperature']
            w_caption = llm_dict['w_caption']
            if args.check_exist_each_iter: # only for multiple classes
                if not cat_exist(
                    pil_img, new_prompt_qkeys_l[0],
                    model, vis_processors, tokenizer,
                    ):
                    return [], []

            cur_image =  np.array(pil_img.convert('RGB'))
            image_height, image_width = cur_image.shape[:2]
            blocks = [(0, 0, image_width, image_height)]
            global text_list_l, text_bg_list
            text_list, bbox_list, bbox_patch_list, caption_list = [], [], [], []
            patches_list = args.patch_list
            all_blocks = []
            for patches in patches_list:
                blocks = [] 
                if patches == 1:
                    blocks.append((0, 0, image_width, image_height))
                elif patches == 0.5:
                    center_left = image_width // 4
                    center_upper = image_height // 4
                    center_right = center_left + (image_width // 2)
                    center_lower = center_upper + (image_height // 2)
                    blocks.extend([(center_left, center_upper, center_right, center_lower)])
                elif patches == 2:
                    mid_width = image_width // 2
                    blocks.extend([(0, 0, mid_width, image_height), (mid_width, 0, image_width, image_height)])

                    mid_height = image_height // 2
                    blocks.extend([(0, 0, image_width, mid_height), (0, mid_height, image_width, image_height)])
                elif patches == 4:
                    blocks.append((0, 0, image_width, image_height))  # 初始块
                    num_cuts = int(np.ceil(np.log2(patches))) 
                    for _ in range(num_cuts):
                        new_blocks = []
                        for left, upper, right, lower in blocks:
                            if (right - left) >= (lower - upper):
                                mid = (left + right) // 2
                                new_blocks.append((left, upper, mid, lower))
                                new_blocks.append((mid, upper, right, lower))
                            else:
                                mid = (upper + lower) // 2
                                new_blocks.append((left, upper, right, mid))
                                new_blocks.append((left, mid, right, lower))
                        blocks = new_blocks
                all_blocks.extend(blocks)  
            if iter == 0:
                text_list_l, text_bg_list = [], []
                for block in all_blocks:
                    left, upper, right, lower = block
                    patch = Image.fromarray(cur_image[upper:lower, left:right]).convert("RGB")
                    text_fg, text_bg, bbox_p, bbox_patch, caption_patch, bbox_avaliable = get_text_from_img_llava_with_bbox_iter0(patch, prompt_q,
                            model, vis_processors, tokenizer,
                            get_bg_text=get_bg_text,
                            conv_mode=conv_mode,
                            temperature=temperature,
                            w_caption=w_caption,
                            use_gene_prompt_fg=use_gene_prompt_fg,
                            reset_prompt_qkeys=reset_prompt_qkeys,
                            new_prompt_qkeys_l=new_prompt_qkeys_l,
                            bg_cat_list=bg_cat_list)
                    bbox_cur_img = convert_patch_bbox_to_original(bbox_p, left, upper)
                    if bbox_avaliable == True:
                        if text_fg not in text_list:
                            text_list.extend(text_fg)
                        text_list_l.extend(text_fg)
                        text_bg_list.extend(text_bg)
                        bbox_list.append(bbox_cur_img)
                        bbox_patch_list.append(bbox_patch)
                        caption_list.append(caption_patch)
                bbox_rel_list = []
                width, height = pil_img.size
                if len(bbox_list) != 0:
                    bbox_most, text_most = bbox_list[0], text_list_l[0]
                    bbox_rel_list = [[round(bbox[0] / width, 2), round(bbox[1] / height, 2), round(bbox[2] / width, 2), round(bbox[3] / height, 2)] for bbox in bbox_list]
                else:
                    bbox_most = [0.0, 0.0, width, height]
                    text_most = text_list_l[0]
                images_with_single_bbox = []
                image_blackout = pil_img.copy()
                draw = ImageDraw.Draw(image_blackout)
                for bbox in bbox_list:
                    draw.rectangle(bbox, fill="black")
                    images_with_single_bbox.append(image_blackout)
                prompt=f"{text_bg_list}, high quality, detailed, blended to the original image."
                negative_prompt=f"{text_list_l}, is a {prompt_qkeys_dict[prompt_q]}"
                seed = 32 # for reproducibility 

                image_black_white = Image.new('RGB', pil_img.size, "black")
                if mask_last != None:
                    mask_array = np.array(mask_last)
                    mask_array[(mask_array[:, :, 0] != 0) | (mask_array[:, :, 1] != 0) | (mask_array[:, :, 2] != 0)] = 255
                    
                    gray = mask_array[:, :, 0]
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_area = 0
                    max_contour = None
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                    x, y, w, h = cv2.boundingRect(max_contour)
                    mask_last = Image.fromarray(mask_array)
                    image_array = np.array(mask_last)
                    if np.all(image_array == 255):
                        generated_image = Image.new('RGB', pil_img.size, (0, 0, 0))
                    else:
                        image_black_white = mask_last
                        draw = ImageDraw.Draw(image_black_white)
                        if image_black_white.mode != 'L':
                            image_black_white = image_black_white.convert('L')
                        generated_image = generate_image(image=pil_img, mask=image_black_white, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, img_path=img_path, seed=seed, iter=iter, sm=sm)
                else:
                    draw = ImageDraw.Draw(image_black_white)
                    for bbox in bbox_list:
                        draw.rectangle(bbox, fill="white")
                    image_array = np.array(image_black_white)
                    if image_black_white.mode != 'L':
                        image_black_white = image_black_white.convert('L')
                    if np.all(image_array == 255):
                        generated_image = Image.new('RGB', pil_img.size, (0, 0, 0))
                    else:
                        generated_image = generate_image(image=pil_img, mask=image_black_white, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, img_path=img_path, seed=seed, iter=iter, sm=sm)
            else:
                images_with_single_bbox = []
                image_blackout = pil_img.copy()
                draw = ImageDraw.Draw(image_blackout)
                for bbox in bbox_list:
                    draw.rectangle(bbox, fill="black")
                    images_with_single_bbox.append(image_blackout)
                prompt=f"{text_bg_list}, high quality, detailed, blended to the original image."
                negative_prompt=f"{text_list_l}, is a {prompt_qkeys_dict[prompt_q]}"
                seed = 32 # for reproducibility 

                image_black_white = Image.new('RGB', pil_img.size, "black")
                if mask_last != None:
                    mask_array = np.array(mask_last)
                    mask_array[(mask_array[:, :, 0] != 0) | (mask_array[:, :, 1] != 0) | (mask_array[:, :, 2] != 0)] = 255
                    
                    gray = mask_array[:, :, 0]
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_area = 0
                    max_contour = None
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                    x, y, w, h = cv2.boundingRect(max_contour)
                    mask_last = Image.fromarray(mask_array)
                    image_array = np.array(mask_last)
                    if np.all(image_array == 255):
                        generated_image = Image.new('RGB', pil_img.size, (0, 0, 0))
                    else:
                        image_black_white = mask_last
                        draw = ImageDraw.Draw(image_black_white)
                        if image_black_white.mode != 'L':
                            image_black_white = image_black_white.convert('L')
                        generated_image = generate_image(image=pil_img, mask=image_black_white, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, img_path=img_path, seed=seed, iter=iter, sm=sm)
                else:
                    draw = ImageDraw.Draw(image_black_white)
                    for bbox in bbox_list:
                        draw.rectangle(bbox, fill="white")
                    image_array = np.array(image_black_white)
                    if image_black_white.mode != 'L':
                        image_black_white = image_black_white.convert('L')
                    if np.all(image_array == 255):
                        generated_image = Image.new('RGB', pil_img.size, (0, 0, 0))
                    else:
                        generated_image = generate_image(image=pil_img, mask=image_black_white, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, img_path=img_path, seed=seed, iter=iter, sm=sm)
                text_list_l, text_bg_list = [], []
                for block in all_blocks:
                    # generate bbox candidte for the image blocks
                    left, upper, right, lower = block
                    patch = Image.fromarray(cur_image[upper:lower, left:right]).convert("RGB")
                    generated_patches = generated_image.crop((left, upper, right, lower))
                    text_fg, text_bg, bbox_p, bbox_patch, caption_patch, bbox_avaliable = get_text_from_img_llava_with_bbox_patch(patch, generated_patches, prompt_q,
                            model, vis_processors, tokenizer,
                            get_bg_text=get_bg_text,
                            conv_mode=conv_mode,
                            temperature=temperature,
                            w_caption=w_caption,
                            use_gene_prompt_fg=use_gene_prompt_fg,
                            reset_prompt_qkeys=reset_prompt_qkeys,
                            new_prompt_qkeys_l=new_prompt_qkeys_l,
                            bg_cat_list=bg_cat_list)
                    bbox_cur_img = convert_patch_bbox_to_original(bbox_p, left, upper)
                    if bbox_avaliable == True:
                        bbox_list.append(bbox_cur_img)
                        bbox_patch_list.append(bbox_patch)
                    if text_fg not in text_list:
                        text_list.extend(text_fg)
                    text_list_l.extend(text_fg)
                    text_bg_list.extend(text_bg)
                    caption_list.append(caption_patch)
                text_most = text_list_l[0]
                bbox_rel_list = []
                if bbox_list != []:
                    bbox_most= bbox_list[0]
                    width, height = pil_img.size
                    bbox_rel_list = [[round(bbox[0] / width, 2), round(bbox[1] / height, 2), round(bbox[2] / width, 2), round(bbox[3] / height, 2)] for bbox in bbox_list]
                else:
                    bbox_most = []

            # generate bbox candidte for the unprocessed full image
            bbox_full = get_reflected_text_from_full_img_llava(pil_img, generated_image, prompt_q, text_list,
                        model, vis_processors, tokenizer,
                        conv_mode=conv_mode,
                        temperature=temperature,
                        reset_prompt_qkeys=reset_prompt_qkeys,
                        new_prompt_qkeys_l=new_prompt_qkeys_l)
            if bbox_full != []:
                bbox_rel_list.append(bbox_full)
            print("instance-specific bounding box", bbox_rel_list)

            # based on previous collected bbox and class infroamtion to get the final text_list and bbox 
            text_final_list, _, bbox, predict_possibility = get_reflected_text_from_img_llava_collected(pil_img, generated_image, bbox_most, bbox_rel_list, prompt_q, text_list, 
                        model, vis_processors, tokenizer,
                        conv_mode=conv_mode,
                        temperature=temperature,
                        reset_prompt_qkeys=reset_prompt_qkeys,
                        new_prompt_qkeys_l=new_prompt_qkeys_l,
                        bg_cat_list=bg_cat_list)    
            return [text_final_list], [text_bg_list], [1.], bbox, predict_possibility

def get_reflected_text_from_img_llava_collected(
    pil_img, image_black, bbox_most, bbox_rel_list, prompt_q, text_candidate_list,
    model, image_processor, tokenizer,
    conv_mode='llava_v0',
    temperature=0.2,
    reset_prompt_qkeys=False,
    new_prompt_qkeys_l=[],
    bg_cat_list=[]):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from contrastive_generate import generate_post
    if reset_prompt_qkeys:
        prompt_qkeys_l = new_prompt_qkeys_l
        question_l = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_l]
        prompt_gene_fg_l = prompt_qkeys_l
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]
        prompt_gene_fg_l = prompt_gene_dict[prompt_q][0]
    text_list = []
    textbg_list = []

    image = pil_img #load_image(img_path)
    image_width, image_height = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # get question index: caption:0, fg:1, bg:2
    fg_idx = 0
    bg_idx = 1
    bounding_box_floats = [0,0,0,0]
    possibility_list = []
    disable_torch_init()
    for qi, qs in enumerate(question_l):
        if qi == 0:
            q_keyword = prompt_qkeys_l[qi]
            caption_q1 = f'The potential {q_keyword} are: {text_candidate_list}, and the potential bounding boxes of {q_keyword} are: {bbox_rel_list}, now you are a detector, output the name of this {q_keyword} in this image in one word.'
            caption_k = f'Output one bounding box of this {q_keyword} in this image.'
            qs = [caption_q1] + [caption_k] 

        image = pil_img #load_image(img_path)
        conv = conv_templates[conv_mode].copy() # 是否需要改一下system 提示词，换成caption？
    
        for i, inp in enumerate(qs):
            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            if i==bg_idx:
                inp = f'The {q_keyword} is {text_list}, and the potential bounding box of {q_keyword} are:{bbox_rel_list}, output only one bounding box to include this {q_keyword} in this image.'
                inp = inp

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            image_sizes=[pil_img.size]
            )

            model_kwargs = {"postion_ids":position_ids,"attention_mask":attention_mask, "inputs_embeds": inputs_embeds}

            image_black_tensor = image_processor.preprocess(image_black, return_tensors='pt')['pixel_values'].half().cuda()

            inputs, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_black_tensor,
                image_sizes=[image_black.size]
            )
            model_kwargs.update({
                    f"position_ids_blackout": position_ids,
                    f"attention_mask_blackout": attention_mask,
                    f"inputs_embeds_blackout": inputs_embeds
                })
            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                None,
                None
            )

            model_kwargs.update( {"postion_ids_blackall":position_ids,"attention_mask_blackall":attention_mask, "inputs_embeds_blackall": inputs_embeds} )

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                generation_output = generate_post(
                    model,
                    input_ids=None,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    streamer=streamer,
                    alpha=1.0,
                    beta=0.0,
                    **model_kwargs,
                    stopping_criteria=[stopping_criteria],
                    return_dict_in_generate=True,
                    output_scores=True)
                
            outputs = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)[0]
            transition_scores = model.compute_transition_scores(generation_output.sequences, generation_output.scores, normalize_logits=True)
            generated_tokens = generation_output.sequences

            if i==fg_idx or i==bg_idx:
                score_list = []
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    if tokenizer.decode(tok) != "[" and tokenizer.decode(tok) != "]" and tokenizer.decode(tok) != '\x00' and tokenizer.decode(tok) != "</s>" and tokenizer.decode(tok) != ",":
                        score_list.append(-score.numpy(force=True))
                possibility_list.append(sum(score_list) / len(score_list))
            conv.messages[-1][-1] = outputs
            outputs_store = outputs
            import re
            outputs_store = re.findall(r'\w+\.?\d*|[^\w\s]', outputs_store)
            if i==bg_idx:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs[-1]=='.':    outputs = outputs[:-1]
                while outputs[0]==' ':  outputs=outputs[1:]

            if i==bg_idx:
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                import re
                outputs = re.sub(r'[\uFEFF\u00A0\u200B\t\n\r\f\v]', ' ', outputs)
                matches = re.findall(r"\[?'?\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\s*\]?'?", outputs, re.DOTALL)
                if matches:
                    bounding_box_values = matches[0]  # This will be a tuple of strings
                    bounding_box_floats = [round(float(value), 3) for value in bounding_box_values]
                else:
                    bounding_box_floats = bbox_most
            elif i==fg_idx:
                text_list.append(outputs)
    bbox_ori = [0,0,0,0]
    if bounding_box_floats != []:
        if bounding_box_floats[0] < 1.0:
            bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = bounding_box_floats[0] * image_width, bounding_box_floats[1] * image_height, bounding_box_floats[2] * image_width, bounding_box_floats[3] * image_height
        elif bounding_box_floats != []:
            bbox_ori = bounding_box_floats
    mean_possibility = sum(possibility_list) / len(possibility_list)
    if len(textbg_list+bg_cat_list)==0:
        textbg_list=['background']
    return text_list, textbg_list+bg_cat_list, bbox_ori, mean_possibility

def convert_patch_bbox_to_original(bounding_box, left, upper):
    x_left, y_top, x_right, y_bottom = bounding_box
    original_x_left = x_left + left
    original_y_top = y_top + upper
    original_x_right = x_right + left
    original_y_bottom = y_bottom + upper
    return [original_x_left, original_y_top, original_x_right, original_y_bottom]




def generate_image(image, mask, prompt, negative_prompt, pipe, img_path, seed, iter, sm):
    # resize for inpainting 
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    generator = torch.Generator('cuda').manual_seed(seed) 

    result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
    resized_result = result.images[0].resize((w, h))
    show(resized_result, f'Contrastive Sample (iter{iter})')

    save_path_dir = '/data/DERI-Gong/jh015/generate_img/'
    img_name = os.path.basename(img_path).split('.')[0]  # Assumes img_path is a path and gets the file name without extension
    os.makedirs(save_path_dir, exist_ok=True) 
    output_path = os.path.join(save_path_dir, f'{img_name}_generate_{iter}.png')
    resized_result.save(output_path)

    return resized_result

def expand_bbox(bbox, expansion_rate=0.15):
    if not bbox or len(bbox) != 4:
        return bbox
    
    x1, y1, x2, y2 = bbox
    
    original_width = x2 - x1
    original_height = y2 - y1
    
    expand_width = original_width * expansion_rate
    expand_height = original_height * expansion_rate
    
    new_x1 = max(0, x1 - expand_width / 2)
    new_y1 = max(0, y1 - expand_height / 2)
    new_x2 = min(1, x2 + expand_width / 2)
    new_y2 = min(1, y2 + expand_height / 2)
    
    return [new_x1, new_y1, new_x2, new_y2]

def get_reflected_text_from_full_img_llava(
    pil_img, image_black, prompt_q, text_candidate_list,
    model, image_processor, tokenizer,
    conv_mode='llava_v0',
    temperature=0.2,
    reset_prompt_qkeys=False,
    new_prompt_qkeys_l=[]):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    # from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from contrastive_generate import generate_post
    if reset_prompt_qkeys:
        prompt_qkeys_l = new_prompt_qkeys_l
        question_l = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_l]
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]

    image = pil_img #load_image(img_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # get question index: caption:0, fg:1, bg:2
    fg_idx = 0
    bg_idx = 1
    bounding_box_floats = [0,0,0,0]
    possibility_list = []
    disable_torch_init()
    for qi, qs in enumerate(question_l):
        if qi == 0:
            q_keyword = prompt_qkeys_l[qi]
            caption_q1 = f'The potential {q_keyword} are: {text_candidate_list}, the ponow you are a detector, output the boundingbox of the {q_keyword}.'
            qs = [caption_q1]

        image = pil_img 
        conv = conv_templates[conv_mode].copy()
    
        for i, inp in enumerate(qs):
            if image is not None:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            image_sizes=[pil_img.size]
            )

            model_kwargs = {"postion_ids":position_ids,"attention_mask":attention_mask, "inputs_embeds": inputs_embeds}

            image_black_tensor = image_processor.preprocess(image_black, return_tensors='pt')['pixel_values'].half().cuda()

            inputs, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_black_tensor,
                image_sizes=[image_black.size]
            )
            model_kwargs.update({
                    f"position_ids_blackout": position_ids,
                    f"attention_mask_blackout": attention_mask,
                    f"inputs_embeds_blackout": inputs_embeds
                })

            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                None,
                None
            )

            model_kwargs.update( {"postion_ids_blackall":position_ids,"attention_mask_blackall":attention_mask, "inputs_embeds_blackall": inputs_embeds} )

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                generation_output = generate_post(
                    model,
                    input_ids=None,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    streamer=streamer,
                    alpha=1.0,
                    beta=0.0,
                    **model_kwargs,
                    stopping_criteria=[stopping_criteria],
                    return_dict_in_generate=True,
                    output_scores=True)
                
            outputs = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)[0]
            transition_scores = model.compute_transition_scores(generation_output.sequences, generation_output.scores, normalize_logits=True)
            generated_tokens = generation_output.sequences

            if i==fg_idx or i==bg_idx:
                score_list = []
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    if tokenizer.decode(tok) != "[" and tokenizer.decode(tok) != "]" and tokenizer.decode(tok) != '\x00' and tokenizer.decode(tok) != "</s>" and tokenizer.decode(tok) != ",":
                        score_list.append(-score.numpy(force=True))
                possibility_list.append(sum(score_list) / len(score_list))
            conv.messages[-1][-1] = outputs
            outputs_store = outputs
            import re
            outputs_store = re.findall(r'\w+\.?\d*|[^\w\s]', outputs_store)
            if i==bg_idx:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') 
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs[-1]=='.':    outputs = outputs[:-1]
                while outputs[0]==' ':  outputs=outputs[1:]
 
            if i==fg_idx:
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') 
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                import re
                outputs = re.sub(r'[\uFEFF\u00A0\u200B\t\n\r\f\v]', ' ', outputs)
                matches = re.findall(r"\[?'?\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\s*\]?'?", outputs, re.DOTALL)
                if matches:
                    bounding_box_values = matches[0] 
                    bounding_box_floats = [round(float(value), 3) for value in bounding_box_values]
                else:
                    bounding_box_floats = [0.0, 0.0, 0.0, 0.0]
    return bounding_box_floats

def get_text_from_img_llava_with_bbox_iter0(
    pil_img, prompt_q,
    model, image_processor, tokenizer,
    get_bg_text=False,
    conv_mode='llava_v0',
    temperature=0.2,
    w_caption=False,
    use_gene_prompt_fg=False,
    reset_prompt_qkeys=False,
    new_prompt_qkeys_l=[],
    bg_cat_list=[]):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    if reset_prompt_qkeys:
        prompt_qkeys_l = new_prompt_qkeys_l
        question_l = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_l]
        prompt_gene_l = [prompt_qkeys_l, ['environment']]
        prompt_gene_fg_l = prompt_qkeys_l
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]
        prompt_gene_l = prompt_gene_dict[prompt_q]
        prompt_gene_fg_l = prompt_gene_dict[prompt_q][0]
    text_list = []
    textbg_list = []

    image = pil_img #load_image(img_path)
    image_width, image_height = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    bbox_avaliable = True
    # get question index: caption:0, fg:1, bg:2
    fg_idx = 0
    bg_idx = 1
    if w_caption:
        fg_idx = 1
        bg_idx = 2

    disable_torch_init()
    for qi, qs in enumerate(question_l):

        if w_caption:
            q_keyword = prompt_qkeys_l[qi]
            bbox_naive = [0,0,0,0]
            caption_q = f'This image is from {q_keyword} detection task, describe the {q_keyword} in one sentence'
            bbox_q = f' The naive bounding box of {q_keyword} is {bbox_naive}, adjust the bounding box to ensure that all {q_keyword} are fully displayed. Just output the adjusted boundingbox.'
            qs=[caption_q] + qs + [bbox_q]


        image = pil_img #load_image(img_path)
        conv = conv_templates[conv_mode].copy() # 是否需要改一下system 提示词，换成caption？

        for i, inp in enumerate(qs):
            if i==fg_idx and use_gene_prompt_fg:
                text_list.append(prompt_gene_fg_l[qi])
                continue
            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            conv.messages[-1][-1] = outputs
            bbox_ori = [0,0,0,0]
            if w_caption and i==0:    
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                caption_list = outputs
            if (w_caption and i==1) or (w_caption and i==2): 
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs[-1]=='.':    outputs = outputs[:-1]
                while outputs[0]==' ':  outputs=outputs[1:]
            if w_caption and i==3:
                import re
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                outputs = outputs.strip('[]</s> \n')
                string_numbers = re.findall(r'\d+\.\d+', outputs)
                outputs_bbox = [round(float(num), 2) for num in string_numbers]
                outputs_bbox = expand_bbox(outputs_bbox, 0.0)
                if not outputs_bbox or len(outputs_bbox) != 4:
                    bbox_avaliable = False
                elif (outputs_bbox[2]-outputs_bbox[0]) * (outputs_bbox[3]-outputs_bbox[1]) == 0:
                    bbox_avaliable = False
                if len(outputs_bbox) >= 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = outputs_bbox[0] * image_width, outputs_bbox[1] * image_height, outputs_bbox[2] * image_width, outputs_bbox[3] * image_height
                bbox_patch = pil_img.crop(bbox_ori)
            if i==fg_idx:
                text_list.append(outputs)
            elif i==bg_idx:
                if outputs.upper() != text_list[-1].upper():
                    textbg_list.append(outputs)

    if len(textbg_list+bg_cat_list)==0:
        textbg_list=['background']
    return text_list, textbg_list+bg_cat_list, bbox_ori, bbox_patch, caption_list, bbox_avaliable

def get_text_from_img_llava_with_bbox_patch(
    pil_img, image_black, prompt_q,
    model, image_processor, tokenizer,
    get_bg_text=False,
    conv_mode='llava_v0',
    temperature=0.2,
    w_caption=False,
    use_gene_prompt_fg=False,
    reset_prompt_qkeys=False,
    new_prompt_qkeys_l=[],
    bg_cat_list=[]):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from contrastive_generate import generate_post
    if reset_prompt_qkeys:
        prompt_qkeys_l = new_prompt_qkeys_l
        question_l = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_l]
        prompt_gene_l = [prompt_qkeys_l, ['environment']]
        prompt_gene_fg_l = prompt_qkeys_l
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]
        prompt_gene_l = prompt_gene_dict[prompt_q]
        prompt_gene_fg_l = prompt_gene_dict[prompt_q][0]
    text_list = []
    textbg_list = []

    image = pil_img #load_image(img_path)
    image_width, image_height = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    fg_idx = 0
    bg_idx = 1
    if w_caption:
        fg_idx = 1
        bg_idx = 2
    bbox_avaliable = True
    disable_torch_init()
    for qi, qs in enumerate(question_l):

        if w_caption:
            q_keyword = prompt_qkeys_l[qi]
            bbox_naive = [0,0,0,0]
            caption_q = f'This image is from {q_keyword} detection task, describe the {q_keyword} in one sentence'
            bbox_q = f' The naive bounding box of {q_keyword} is {bbox_naive}, adjust the bounding box to ensure that all {q_keyword} are fully displayed. Just output the adjusted boundingbox.'
            qs=[caption_q] + qs + [bbox_q]


        image = pil_img 
        conv = conv_templates[conv_mode].copy() 

        for i, inp in enumerate(qs):
            if i==fg_idx and use_gene_prompt_fg:
                text_list.append(prompt_gene_fg_l[qi])
                continue
            if image is not None:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            image_sizes=[pil_img.size]
            )

            model_kwargs = {"postion_ids":position_ids,"attention_mask":attention_mask, "inputs_embeds": inputs_embeds}

            image_black_tensor = image_processor.preprocess(image_black, return_tensors='pt')['pixel_values'].half().cuda()

            inputs, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_black_tensor,
                image_sizes=[image_black.size]
            )
            model_kwargs.update({
                    f"position_ids_blackout": position_ids,
                    f"attention_mask_blackout": attention_mask,
                    f"inputs_embeds_blackout": inputs_embeds
                })
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                generation_output = generate_post(
                    model,
                    input_ids=None,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    streamer=streamer,
                    alpha=1.0,
                    beta=0.0,
                    **model_kwargs,
                    stopping_criteria=[stopping_criteria],
                    return_dict_in_generate=True,
                    output_scores=True)
            outputs = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)[0]
            conv.messages[-1][-1] = outputs
            bbox_ori = [0,0,0,0]
            if w_caption and i==0:    
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                caption_list = outputs
            if (w_caption and i==1) or (w_caption and i==2): 
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs[-1]=='.':    outputs = outputs[:-1]
                while outputs[0]==' ':  outputs=outputs[1:]
            if w_caption and i==3:
                import re
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                outputs = outputs.strip('[]</s> \n')
                string_numbers = re.findall(r'\d+\.\d+', outputs)
                outputs_bbox = [round(float(num), 2) for num in string_numbers]
                if not outputs_bbox or len(outputs_bbox) != 4:
                    bbox_avaliable = False
                if len(outputs_bbox) >= 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = outputs_bbox[0] * image_width, outputs_bbox[1] * image_height, outputs_bbox[2] * image_width, outputs_bbox[3] * image_height
                bbox_patch = pil_img.crop(bbox_ori)
            if i==fg_idx:
                text_list.append(outputs)
                if not get_bg_text: break
            elif i==bg_idx:
                if outputs.upper() != text_list[-1].upper():
                    textbg_list.append(outputs)

    if len(textbg_list+bg_cat_list)==0:
        textbg_list=['background']
    return text_list, textbg_list+bg_cat_list, bbox_ori, bbox_patch, caption_list, bbox_avaliable

def clip_similarity(patch_img_list, masks_list, sm_list, masks_weight_list, text, model, device="cuda"):
    images = []
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    for ith in range(len(patch_img_list)):
        images.append(preprocess(patch_img_list[ith]))
    image = torch.tensor(np.stack(images)).to(device)
    text = clip.tokenize(text).to(device) # Tokenize the Text with CLIP
    with torch.no_grad():
        logits_per_image, _ = model(image, text) # Pass both text and Image as Input to the model
        similarity = logits_per_image.softmax(dim=0)
        if len(similarity) > 1:
            similarity = (similarity - similarity.min() + 1e-9 ) / (similarity.max() - similarity.min()+ 1e-9)
        similarity_sum = similarity
        indices = (similarity_sum > 0.7).nonzero(as_tuple=True)[0].tolist()
        if 0 not in indices:
            indices.append(0)
        
        weighted_mask_list = masks_weight_list
        if len(similarity) > 1:
            weight_list = similarity_sum
            total_weight = weight_list.sum().item()
            weight_list = (weight_list.cpu() / total_weight).numpy()
            weighted_mask_list = np.array(weighted_mask_list)
            weighted_mask_list = weight_list.reshape((len(similarity), 1, 1)) * weighted_mask_list
            sm_list = np.array(sm_list)
            sm_list = weight_list.reshape((len(similarity), 1, 1)) * sm_list
            weighted_sum = np.zeros_like(weighted_mask_list[0]).astype(np.float32)
            for i in range(len(weighted_mask_list)):
                weighted_sum += weighted_mask_list[i]
            min_val = weighted_sum.min()
            max_val = weighted_sum.max()
            normalized_weighted_mask = (weighted_sum - min_val + 1e-9) / (max_val - min_val + 1e-9)
            weighted_sum[weighted_sum >= 1] = 1
            weighted_sum_sm = np.zeros_like(sm_list[0]).astype(np.float32)
            for i in range(len(sm_list)):
                weighted_sum_sm += sm_list[i]
            min_val = weighted_sum_sm.min()
            max_val = weighted_sum_sm.max()
            normalized_weighted_sm = (weighted_sum_sm - min_val + 1e-9) / (max_val - min_val + 1e-9)
        else:
            normalized_weighted_mask = masks_list[0]
            weighted_sum = masks_list[0]
            normalized_weighted_sm = [1.0]

        select_img = [patch_img_list[i] for i in indices]
        selcet_mask = [masks_list[i] for i in indices]
        select_weight = [sm_list[i] for i in indices]
        if isinstance(select_img, list):
            base_image = Image.new('RGB', select_img[0].size, 'black')
            mask_height, mask_width = selcet_mask[0].shape[:2]
            mask_image = Image.new('L', (mask_width, mask_height), 'black')
            ### image
            for img in select_img:
                np_img = np.array(img)
                np_base = np.array(base_image)
                np_base[np_img != 0] = np_img[np_img != 0]
                base_image = Image.fromarray(np_base)
            ### mask
            for np_mask in selcet_mask:
                np_base_mask = np.array(mask_image)
                np_base_mask[np_mask != 0] = np_mask[np_mask != 0]
                mask_image = Image.fromarray(np_base_mask)

        else:
            #### image
            base_image = Image.new('RGB', select_img.size, 'black')
            mask_height, mask_width = selcet_mask.shape[:2]
            mask_image = Image.new('L', (mask_width, mask_height), 'black')
            np_img = np.array(select_img)
            np_base = np.array(base_image)
            np_base[np_img != 0] = np_img[np_img != 0]
            base_image = Image.fromarray(np_base)
            #### mask
            np_mask = selcet_mask
            np_base_mask = np.array(mask_image)
            np_base_mask[np_mask != 0] = np_mask[np_mask != 0]
            normalized_weighted_sm = [1.0]
            
        return np_base_mask, normalized_weighted_mask, weighted_sum, normalized_weighted_sm
