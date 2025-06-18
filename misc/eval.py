import torch
import torch.nn.functional as F
# import clip
from text_utils.tokenizer import tokenize
from functools import partial
from prettytable import PrettyTable
import numpy as np
import os

def write_txt(data, name):
    with open(name, 'a') as f:
        f.write(data)
        f.write('\n')

@torch.no_grad()
def pretrain_val(model, gallery_dataloader, query_dataloader, config):
    # switch to evaluate mode
    model.eval()

    gallery_dataset = gallery_dataloader.dataset
    query_dataset = query_dataloader.dataset

    text_feats = []
    # composed_feats = []
    sketch_feats = []

    for text, sketch, text_with_blank in query_dataloader:
        text_tokens = tokenize(text, context_length=config.experiment.text_length)
        text_tokens = text_tokens.squeeze().cuda()

        text_feat = F.normalize(model.encode_text(text_tokens), dim=-1)
        text_feats.append(text_feat)
        sketch = sketch.cuda()
        sketch_feat = model.encode_image(sketch)

        sketch_feat_norm = F.normalize(sketch_feat, dim=-1)
        sketch_feats.append(sketch_feat_norm)    

    image_feats = []

    for image in gallery_dataloader:
        # image = image.to(device)
        image = image.cuda()
        image_feat = model.encode_image(image)
        image_feat_norm = F.normalize(image_feat, dim=-1)
        image_feats.append(image_feat_norm)

    query_feats = {
        'textonly2rgb': torch.cat(text_feats, dim=0),
        'sketchonly2rgb': torch.cat(sketch_feats, dim=0),
    }

    metric_func = partial(metric_eval, 
                            gally_feats=torch.cat(image_feats, dim=0),      #gallery
                            gallerylabel = gallery_dataset.img2person
                            )
    
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])

    for key, value in query_feats.items():
        eval_result  = metric_func(query_feats=value, querylabel = query_dataset.txt2person, set = 0)        
        if(key == "sketchonly2rgb"):
            # Sketch retrieval testing for Tri-PEDES
            eval_result  = metric_func(query_feats=value, querylabel = query_dataset.txt2person, g_camids = gallery_dataset.cam_id, q_camids = query_dataset.cam_id, set = 2)  
            save_r1 = eval_result['r1']

        table.add_row([key, eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP'], eval_result['mINP']])

    table.float_format = '.4'

    return table, save_r1


@torch.no_grad()
def fintune_val(model, gallery_dataloader, query_dataloader, config):
    # switch to evaluate mode
    model.eval()

    gallery_dataset = gallery_dataloader.dataset
    query_dataset = query_dataloader.dataset

    text_feats = []
    query_text_list = []
    composed_feats = []
    mixture_feats = []
    mixture_token_feats = []
    sketch_feats = []
    sketch_token_feats = []
    sketch_feats_out = []
    mixture_feats_out = []
    #一个id一个对应的text和sketch,共50个query
    for text, sketch, text_with_blank in query_dataloader:
        text_tokens = tokenize(text, context_length=config.experiment.text_length)
        text_tokens = text_tokens.squeeze().cuda()
        query_text_list = query_text_list + list(text)
        text_with_blank_tokens = tokenize(text_with_blank, context_length=config.experiment.text_length)
        text_with_blank_tokens = text_with_blank_tokens.squeeze().cuda()
        text_feat = F.normalize(model.tbps_clip.encode_text(text_tokens), dim=-1)
        text_feats.append(text_feat)

        sketch = sketch.cuda()
        sketch_feat = model.tbps_clip.encode_image(sketch)

        mixture_feat = text_feat + sketch_feat
        mixture_feat = F.normalize(mixture_feat, dim=-1)
        mixture_feats.append(mixture_feat)
        
        sketch_feat_norm = F.normalize(sketch_feat, dim=-1)
        sketch_feats.append(sketch_feat_norm)

        sketch_token = model.mapping(sketch_feat)
        # sketch_token = sketch_feat
        composed_feat = model.tbps_clip.encode_text.encode_prompt_token(text_with_blank_tokens, sketch_token).float()  
        composed_feat_norm = F.normalize(composed_feat, dim=-1)
        composed_feats.append(composed_feat_norm)

        sketch_token_feat =model.get_text_pseudo_feat(sketch_feat)
        mixture_token_feat = text_feat + sketch_token_feat
        mixture_token_feat = F.normalize(mixture_token_feat, dim=-1)
        mixture_token_feats.append(mixture_token_feat)
        sketch_token_norm = F.normalize(sketch_token_feat, dim=-1)
        sketch_token_feats.append(sketch_token_norm)

    image_feats = []
    for image in gallery_dataloader:
        image = image.cuda()
        image_feat = model.tbps_clip.encode_image(image)
        image_feat_norm = F.normalize(image_feat, dim=-1)
        image_feats.append(image_feat_norm)

    query_feats = {
        'interactive_retrieval': torch.cat(composed_feats, dim =0),
        # 'sketchtoken2rgb': torch.cat(sketch_token_feats, dim=0),
        # 'mixture_token2rgb': torch.cat(mixture_token_feats, dim=0),
        'sketchonly_retrieval': torch.cat(sketch_feats, dim=0),
        # 'textonly2rgb': torch.cat(text_feats, dim=0),
        # 'mixtureonly2rgb': torch.cat(mixture_feats, dim=0),
    }

    metric_func = partial(metric_eval, 
                            gally_feats=torch.cat(image_feats, dim=0),      #gallery
                            gallerylabel = gallery_dataset.img2person
                            )
    
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])

    for key, value in query_feats.items():
        eval_result  = metric_func(query_feats=value, querylabel = query_dataset.txt2person, set = 0)        
        if(key == "interactive_retrieval"):
            save_r1 = eval_result['r1']
        table.add_row([key, eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP'], eval_result['mINP']])
    table.float_format = '.4'

    # Visualization log generation

    # if(config.model.ckpt_type == 'direct_test'):
        
    #     rank_visualize_func = partial(rank_visualize, 
    #                             gally_feats=torch.cat(image_feats, dim=0),    
    #                             gallerylabel = gallery_dataset.img2person,
    #                             query_text = query_text_list, 
    #                             sketch_querypath = query_dataset.file_path,
    #                             gallery_path = gallery_dataset.file_path
    #                             )                                
    #     for key, value in query_feats.items():
    #         if(key == "composed2rgb" or key == "sketchonly2rgb" or key == "mixtureonly2rgb"):
    #             visualize_log = os.path.join(config.model.output_path, "visualize_{}_log.txt".format(key))
    #             write_txt(f"-----------------Query type:{key}------------------\n", visualize_log)                
    #             rank_visualize_func(query_feats=value, querylabel=query_dataset.txt2person, visualize_log=visualize_log)

    return table, save_r1


@torch.no_grad()
def metric_eval(query_feats, gally_feats, querylabel, gallerylabel, q_camids=torch.tensor([]), g_camids=torch.tensor([]), set = 0):
    scores_t2i = query_feats @ gally_feats.t()

    device = scores_t2i.device
    gallerylabel = gallerylabel.cuda()  #[num_g]
    querylabel = querylabel.cuda()  #[num_q]

    #返回对scores_t2i的最后一维按照降序排列的索引值，index:[num_q, num_g]
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = gallerylabel[index]   #[num_g]
    matches = (querylabel.view(-1, 1).eq(pred_person))  # [num_q, num_g]
    num_q, num_g = matches.size()
    if(set ==2):    
        q_camids = q_camids.cuda()  #[num_q]
        g_camids = g_camids.cuda()  #[num_g]

        cam_remove = (querylabel.view(-1, 1).eq(gallerylabel[index])) & (q_camids.view(-1, 1).eq(g_camids[index]))
        # [num_q, num_g]
        cam_keep = ~cam_remove
        # matches = matches & cam_keep  # [num_q, num_g]
        matches = torch.masked_select(matches, cam_keep)
        matches = matches.view(num_q, num_g - 1)  # [num_q, num_g-1]

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = AP.nanmean() * 100.0

    dim0, dim1 = matches.shape
    cmc = matches.cumsum(dim=-1).float()
    mINP = []
    for q_idx in range(dim0):
        matches_q = matches[q_idx]
        cmc_q = cmc[q_idx]
        pos_idx = torch.where(matches_q==1)[0]
        if(pos_idx.numel() == 0):
            continue
        pos_max_idx = torch.max(pos_idx)
        INP = cmc_q[pos_max_idx] / (pos_max_idx+1.0)
        mINP.append(INP)
    mINP = torch.stack(mINP, dim=0)
    mINP = mINP.mean() * 100.0
    
    eval_result = {'r1': ir1,
                   'r5': ir5,
                   'r10': ir10,
                   'r_mean': ir_mean,
                   'mAP': mAP.item(),
                   'mINP': mINP.item()
                   }
    return eval_result

def rank_visualize(query_feats, gally_feats, querylabel, gallerylabel, query_text, sketch_querypath, gallery_path, visualize_log):
    cos_sim = query_feats @ gally_feats.t()
    distmat = 1 - cos_sim
    distmat = distmat.cpu().numpy()
    querylabel = querylabel.detach().cpu().numpy()
    gallerylabel = gallerylabel.detach().cpu().numpy()
    num_q, num_g = distmat.shape
    gallery_indices = np.argsort(distmat, axis=1)  #[num_q, num_g]  返回相似度从大到小排序后的索引.即与query最相似的gallery index在前面。 
    # argsort默认升序，距离约小越靠前
    gallery_indices_top10 = gallery_indices[:, :10]

    for q_idx in range(num_q):  #遍历每个query 
        write_txt(f"----------------------------------------------------- \n", visualize_log)
        write_txt(f"query index: {q_idx}, sketch_query_label: {querylabel[q_idx]}, query_text: {query_text[q_idx]}, query_sketch_path: {sketch_querypath[q_idx]} \n ", visualize_log)

        g_idx_top10 = gallery_indices_top10[q_idx]        #(10, )
        for g_idx in range(10):
            gg = g_idx_top10[g_idx]
            cos_sim = -distmat[q_idx][gg]
            write_txt(f"matched gallery index: {gg}, matched_gallery_label:{gallerylabel[gg]}, cosine_sim:{cos_sim}, gallery_img_path: {gallery_path[gg]} \n", visualize_log)

    return 0