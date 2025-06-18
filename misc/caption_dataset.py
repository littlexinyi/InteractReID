import json
import os
import re
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random
# from text_utils.tokenizer import tokenize

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class Choose:
    def __init__(self, rand_from, size):
        self.choose_from = rand_from
        self.size = size

    def __call__(self, image):
        aug_choice = np.random.choice(self.choose_from, 2)
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            *aug_choice,
            normalize
        ])(image)



def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
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

class ps_train_dataset(Dataset):
    def __init__(self, ann_root, transform, aug_ss, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file))
        self.transform = transform

        image_root = os.path.join(ann_root, 'imgs/')

        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)

        self.augmentation_ss = aug_ss

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, caption_bt, person = self.pairs[index]
        sketch_path = image_path.replace('imgs', 'imgs-sketch')

        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        aug1 = self.transform(image_pil.convert('RGB'))
        aug_ss_1 = self.augmentation_ss(image_pil)
        aug_ss_2 = self.augmentation_ss(image_pil)

        sketch_pil = Image.open(sketch_path)
        sketch = self.transform(sketch_pil.convert('RGB'))   

        # caption = tokenize(caption, context_length = 77)
        # caption_bt = tokenize(caption_bt, context_length = 77)

        return {
            'image': image,
            'sketch': sketch,
            'caption': caption,
            'caption_bt': caption_bt,
            'id': person,
            'aug1': aug1,
            'aug_ss_1': aug_ss_1,
            'aug_ss_2': aug_ss_2
        }


class ps_eval_img_dataset(Dataset):
    def __init__(self, ann_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform

        image_root = os.path.join(ann_root, 'imgs/')

        # self.text = []
        self.image = []
        # self.text_img = []
        # self.txt2person = []
        self.img2person = []
        self.cam_id = []
        id = 1
        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            self.image.append(image_path)

            person_id = ann['id']
            self.img2person.append(person_id)
            self.cam_id.append(id)
            id +=1
            # for caption in ann['captions']:
            #     self.text.append(pre_caption(caption, max_words))
            #     self.txt2person.append(person_id)
            #     self.text_img.append(image_path)

        # self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)
        self.cam_id = torch.tensor(self.cam_id, dtype=torch.long)
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        # sketch_path = image_path.replace('imgs', 'imgs-sketch')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # sketch = Image.open(sketch_path).convert('RGB')
        # sketch = self.transform(sketch)
        # return image, sketch
        return image

class ps_eval_text_dataset(Dataset):
    def __init__(self, ann_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform

        image_root = os.path.join(ann_root, 'imgs/')
        self.text = []
        self.text_img = []
        self.txt2person = []
        self.cam_id = []
        id = 1
        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, max_words))
                self.txt2person.append(person_id)
                self.text_img.append(image_path)
                self.cam_id.append(id)
            id += 1

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.cam_id = torch.tensor(self.cam_id, dtype=torch.long)
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        image_path = self.text_img[index]
        sketch_path = image_path.replace('imgs', 'imgs-sketch')

        sketch = Image.open(sketch_path).convert('RGB')
        sketch = self.transform(sketch)
        text = self.text[index]

        # text_tokens = tokenize(text, context_length = 77)
        text_with_blank = 'a photo of * , and {}'.format(text)
        # text_with_blank_tokens = tokenize(text_with_blank, context_length = 77)

        return text, sketch, text_with_blank
        # return text, text_with_blank



class pedes_train_dataset(Dataset):
    def __init__(self, ann_root1, ann_root2, ann_root3, transform, aug_ss, split, max_words=30):

        #ann_root1: '/data0/wxy_data/datasets/CUHK-PEDES/CUHK-PEDES' 
        #ann_root2: '/data0/wxy_data/datasets/ICFG-PEDES/ICFG-PEDES'
        #ann_root3: '/data0/wxy_data/datasets/RSTPReid/RSTPReid'

        ann_file1 = os.path.join(ann_root1, split + '_reid.json')
        anns_1 = json.load(open(ann_file1, 'r'))
        image_root1 = os.path.join(ann_root1, 'imgs/')

        ann_file2 = os.path.join(ann_root2, split + '_reid.json')
        anns_2 = json.load(open(ann_file2, 'r'))
        image_root2 = os.path.join(ann_root2, 'imgs/')

        ann_file3 = os.path.join(ann_root3, split + '_reid.json')
        anns_3 = json.load(open(ann_file3, 'r'))
        image_root3 = os.path.join(ann_root3, 'imgs/')

        self.transform = transform
        self.augmentation_ss = aug_ss

        self.person2text = defaultdict(list) #每个行人ID的所有caption为一个项
        
        self.person_id2idx = {}
        n = 0
        self.pairs = []
        #遍历CUHK-train_reid.json
        for ann in anns_1:
            image_path = os.path.join(image_root1, ann['file_path'])
            person_id = ann['id']
            person_id = 'CUHK' + str(person_id)
            if person_id not in self.person_id2idx.keys():  
                 #新的ID出现，则添加到ID字典中
                self.person_id2idx[person_id] = n
                n += 1
            person_idx = self.person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)

        for ann in anns_2:
            image_path = os.path.join(image_root2, ann['file_path'])
            person_id = ann['id']
            person_id = 'ICFG' + str(person_id)
            if person_id not in self.person_id2idx.keys():  
                 #新的ID出现，则添加到ID字典中
                self.person_id2idx[person_id] = n
                n += 1
            person_idx = self.person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)

        for ann in anns_3:
            image_path = os.path.join(image_root3, ann['file_path'])
            person_id = ann['id']
            person_id = 'RSTP' + str(person_id)
            if person_id not in self.person_id2idx.keys():  
                 #新的ID出现，则添加到ID字典中
                self.person_id2idx[person_id] = n
                n += 1
            person_idx = self.person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)

        # print("person_id2idx:", len(self.person_id2idx))      #17806
        # print("person2text:", len(self.person2text))    #17806
        # print("pairs:", len(self.pairs))    #139810

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, caption_bt, person = self.pairs[index]
        sketch_path = image_path.replace('imgs', 'imgs-sketch')

        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        aug1 = self.transform(image_pil.convert('RGB'))
        aug_ss_1 = self.augmentation_ss(image_pil)
        aug_ss_2 = self.augmentation_ss(image_pil)

        sketch_pil = Image.open(sketch_path)
        sketch = self.transform(sketch_pil.convert('RGB'))   

        # caption = tokenize(caption, context_length = 77)
        # caption_bt = tokenize(caption_bt, context_length = 77)

        return {
            'image': image,
            'sketch': sketch,
            'caption': caption,
            'caption_bt': caption_bt,
            'id': person,
            'aug1': aug1,
            'aug_ss_1': aug_ss_1,
            'aug_ss_2': aug_ss_2
        }





class MaSk1K_train(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.files_rgb = os.listdir(data_path+'/photo/train')
        self.files_sk = {s: os.listdir(f'{self.data_path}/sketch/{s}/train') for s in 'ABCDEF'}
        self.person2text = set()   #无重复且按照升序排列
        self.sketch_pid_all = []
        self.path_sk = []
        for s in self.files_sk.keys():  #分别遍历每一个风格     #根据sketch的Person ID构建训练集的label
            files = self.files_sk[s]
            for img_path in files:
                self.path_sk.append(self.data_path+'/sketch/'+s+'/train/'+img_path) #sketch_train 的所有路径列表

                pid = int(img_path[:4])
                self.sketch_pid_all.append(pid)
                self.person2text.add(pid)  #不重复的pid集合            type = set 
        self.pid2label = {pid:label for label, pid in enumerate(self.person2text)}  #label为对pid的重排序编码

        self.sketch_label_all = [self.pid2label[pid] for pid in self.sketch_pid_all]

        #pid2label: {pid:label, ...}
        self.path_rgb = {}   
        for pid in self.pid2label.keys():
            self.path_rgb[pid] = [os.path.join(self.data_path, 'photo', 'train', img_path) for img_path in self.files_rgb if int(img_path[:4]) == pid]

    def __len__(self):
        return len(self.path_sk)
    
    def __getitem__(self, index):

        sk_filepath = self.path_sk[index]
        sk_pid = os.path.basename(sk_filepath)[:4]
        sk_pid = int(sk_pid)
        label = self.pid2label[sk_pid]       
        rgb_path = np.random.choice(self.path_rgb[sk_pid])

        sketch = Image.open(sk_filepath).convert('RGB')
        sketch = self.transform(sketch)

        image = Image.open(rgb_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image, 
            'sketch': sketch, 
            'id': label}
    
class Mask1K_test_img(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.files = os.listdir(self.data_path+'/photo/query')  
        self.file_path = []
        self.img2person = []        

        for file in self.files:
            self.file_path.append(os.path.join(self.data_path, 'photo/query', file))
            self.img2person.append(int(file[:4]))      

        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        img_path = self.file_path[index]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image


from scipy.io import loadmat

def read_attributes(data_path):
    tmp = [[],[]]
    names = ['gender', 'hair', 'up', 'down', 'clothes', 'hat', 'backpack', 'bag', 'handbag', 'age',\
        'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen',\
        'downblack', 'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']

        # save all attribute -> (1501, 27)  #1501个人，每个人都只有一个attribute标注
    mat = loadmat(f'{data_path}/market_attribute.mat')['market_attribute']    #numpy
        # print(mat)
    newM = np.zeros((27,1502))
    for i in range(751):
        m = mat[0][0][1][0][0]
        for j in range(27):
            newM[j][int(m[27][0][i])] = m[names[j]][0][i]
        tmp[0].append(int(m[27][0][i]))

    for i in range(750):
        m = mat[0][0][0][0][0]
        for j in range(27):
            newM[j][int(m[27][0][i])] = m[names[j]][0][i]
        tmp[1].append(int(m[27][0][i]))
        # print(newM.shape)             #[27,1502]
        # print(self.pid2label)
        # save train attribute and relabel
        # trainM = np.zeros((len(self.pid2label),27))
        # for id,l in self.pid2label.items():
        #     trainM[l] = newM.T[id]          #[498,27]  numpy
    newM = newM.T               #[1502, 27]
    return newM

def get_textInput(idx):
    # idx: (27)
    all_attributes = []
    gender = ['person', 'man', 'woman'][idx[0]]
    all_attributes.append(gender)
    annun = ['He or she', 'He', 'She'][idx[0]]
    hairLen = ['', 'short', 'long'][idx[1]]
    all_attributes.append(hairLen+" hair")
    sleeveLen = ['', 'long', 'short'][idx[2]]
   
    lowerLen = ['', 'long', 'short'][idx[3]]
    lowerType = ['clothes', 'dress', 'pants'][idx[4]]
    
    accesories = ' with ' + ' and '.join(np.array(['a hat', 'a backpack', 'a bag', 'a handbag'])[np.where(idx[5:9]==2)]) + '.' if 2 in idx[5:9] else '.'
    accesory_attri = ' with ' + ' and '.join(np.array(['a hat', 'a backpack', 'a bag', 'a handbag'])[np.where(idx[5:9]==2)]) + '.' if 2 in idx[5:9] else ''
    if(accesory_attri != ''):
        all_attributes.append(accesory_attri)
    elif(accesory_attri == ''):
        all_attributes.append('with no accesory')
    age = ['', 'young', 'teenager', 'adult', 'old'][idx[9]]
    all_attributes.append(age)
    person = age +" " + gender
    upColorAll = np.array(['black', 'white', 'red', 'purple', 'yellow', 'gray', 'blue', 'green'])
    upColor = ' and '.join(upColorAll[np.where(idx[10:18]==2)]) + ' ' if 2 in idx[10:18] else ''
    all_attributes.append(upColor + sleeveLen + " sleeve clothes")     #TODO 可增加详细描述的属性, jacket T-shirt?
    # upColor = ' and '.join(upColorAll[np.where(idx[10:18]==2)])
    downColorAll = np.array(['black', 'white', 'pink', 'purple', 'yellow', 'gray', 'blue', 'green', 'brown'])
    downColor = ' and '.join(downColorAll[np.where(idx[18:]==2)]) +  ' ' if 2 in idx[18:] else ''
    all_attributes.append(downColor + lowerLen +" "+ lowerType)
    answer = ' '.join(['An' if idx[9] in [3,4] else 'A', person, 'with', hairLen, 'hair.', annun, 'is in', upColor, sleeveLen,'sleeve clothes, and wears',\
         downColor, lowerLen, lowerType]) + accesories

    # print("all_attributes:", attributes_str)
    return answer.replace('  ',' ').replace('  ',' ').replace('  ',' '), all_attributes



class Mask1K_test_sk(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.files = {s: os.listdir(f'{self.data_path}/sketch/{s}/query') for s in 'ABCDEF'}
        self.file_path = []
        self.txt2person = []

        for s in self.files.keys():
            files = self.files[s]
            for img_path in files:
                self.file_path.append(os.path.join(self.data_path, 'sketch/',s,'query/', img_path))
                self.txt2person.append(int(img_path[:4]))

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)

        self.attribute_numpy = read_attributes(self.data_path).astype(int) 

        # text_file = os.path.join(self.data_path, 'text_description.json')
        # text_anns = json.load(open(text_file))
        # self.captions_dict = defaultdict(list)
        # for obj in text_anns:
        #     self.captions_dict[obj['id']].append(obj['caption'])     #{id1:[caption1, caption2], id2:[caption1]}


    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        img_path = self.file_path[index]
        label = self.txt2person[index]
        sketch = Image.open(img_path).convert('RGB')
        sketch = self.transform(sketch)

        attribute_numpy = self.attribute_numpy[label]   
        caption, all_attributes_list = get_textInput(attribute_numpy)
        #TODO 建立一个label/id为索引的字典，直接索引到id对应的caption
        # captions_list = self.captions_dict[label.item()]
        # caption = random.choice(captions_list)

        text_with_blank = 'a photo of * , and {}'.format(caption)

        return caption, sketch, text_with_blank
    

from PIL import Image, ImageOps
def re_label(lables):
    pid2label = {pid:label for label, pid in enumerate(np.sort(np.unique(lables)))}
    new_labels = [pid2label[label] for label in lables]
    return new_labels

class PKUSketch_Train(Dataset):
    def __init__(self, data_path, index_id, transform):
        self.transform = transform
        self.data_path = data_path
        # self.data_path = '/data0/wxy_data/datasets/PKUSketch/'
        self.all_files_rgb = os.listdir(self.data_path+'/photo')  #所有的train image file目录    xxxxx.jpg,但返回的排序混乱
        self.all_files_sk = os.listdir(self.data_path + '/sketch')
        self.ind_list = np.sort(index_id)
        self.selected_path_rgb = []
        self.selected_path_sk = []
        self.person2text = set() 
        self.labels = []
        f = open(self.data_path + '/TriReID.txt', 'r')
        self.txt_list = f.readlines()       # 按顺序返回
        self.selected_txt = []
        for idx in self.ind_list:   #按照一定比例随机选择的index
            for img_path in self.all_files_rgb:
                idx_path = int(img_path.split('/')[-1].split('_')[0])
                if idx_path == idx: #符合预期行人ID，则加入
                    self.selected_path_rgb.append(os.path.join(self.data_path,'photo/', img_path))
                    
            for img_path in self.all_files_sk:
                idx_path = int(img_path.split('/')[-1].split('.')[0])
                if idx_path == idx: 
                    self.selected_path_sk.append(os.path.join(self.data_path,'sketch/', img_path))
            
            for txt_description in self.txt_list:
                idx_path = int(txt_description.split('/')[0])
                if idx_path == idx:
                    txt1 = txt_description.split('\n')[0]
                    txt = txt1.split('@')[-1]
                    self.selected_txt.append(txt)
            
            self.labels.append(idx)
            self.person2text.add(idx)

        self.labels = re_label(self.labels)

    def __len__(self):
        return len(self.selected_path_sk)
    
    def __getitem__(self, index): 
        
        img_path =  self.selected_path_rgb[index]
        img_data = Image.open(img_path).convert('RGB')
        image = self.transform(img_data)
        
        text = self.selected_txt[index]

        sk_path = self.selected_path_sk[index // 2]
        sk_data = Image.open(sk_path).convert('RGB')
        sketch = self.transform(sk_data)
        
        label = self.labels[index // 2]
        
        return {
            'image': image, 
            'sketch': sketch, 
            'id': label,
            'text': text}


class PKUSketch_Test_img(Dataset):
    def __init__(self, data_path, index_id, transform):
        self.transform = transform
        self.data_path = data_path
        self.ind_list = np.sort(index_id)
        self.selected_path_rgb = []
        self.img2person = []
        # f = open(self.data_path + '/TriReID.txt', 'r')
        # self.txt_list = f.readlines()       # 按顺序返回

        self.all_files = os.listdir(self.data_path+'/photo')
        for idx in self.ind_list:
            for img_path in self.all_files:
                idx_path = int(img_path.split('/')[-1].split('_')[0])
                if idx_path == idx: 
                    self.selected_path_rgb.append(os.path.join(self.data_path,'photo/', img_path))
                    self.img2person.append(idx)

        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __getitem__(self, index):
  
        img_path =  self.selected_path_rgb[index]
        img_data = Image.open(img_path).convert('RGB')
        image = self.transform(img_data)
        
        return image

    def __len__(self):
        return len(self.selected_path_rgb)

#一个ID对应2个rgb img, 2个相同的text文本，1个sketch

class PKUSketch_Test_query(Dataset):
    def __init__(self, data_path, index_id, transform):
        self.transform = transform
        self.data_path = data_path
        # self.data_path = '/data0/wxy_data/datasets/PKUSketch/'
        self.ind_list = np.sort(index_id)
        self.selected_path_sk = []

        f = open(self.data_path + '/TriReID.txt', 'r')
        self.txt_list = f.readlines()       # 按顺序返回
        self.selected_txt = []
        self.txt2person = []
        self.all_files = os.listdir(self.data_path+'/sketch')
        for idx in self.ind_list:
            for img_path in self.all_files:
                idx_path = int(img_path.split('/')[-1].split('.')[0])
                if idx_path == idx: 
                    self.selected_path_sk.append(os.path.join(self.data_path,'sketch/', img_path))
                    self.txt2person.append(idx)            

            for txt_description in self.txt_list:
                idx_path = int(txt_description.split('/')[0])
                if idx_path == idx:
                    txt1 = txt_description.split('\n')[0]
                    txt = txt1.split('@')[-1]
                    self.selected_txt.append(txt)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)

    def __getitem__(self, index):
  
        sketch_path =  self.selected_path_sk[index]
        sketch_data = Image.open(sketch_path).convert('RGB')
        sketch = self.transform(sketch_data)       

        text1 = self.selected_txt[int(index*2)]
        text2 = self.selected_txt[int(index*2 + 1)]
        text = random.choice([text1, text2])
        
        text_with_blank = 'a photo of * , and {}'.format(text)
        
        return text, sketch, text_with_blank
        # return text, text_with_blank

    def __len__(self):
        return len(self.selected_path_sk)


class QMUL_Shoe_Train(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        train_visible_path = os.path.join(self.data_path, 'ShoeV2/train_visible.txt')
        train_sketch_path = os.path.join(self.data_path, 'ShoeV2/train_sketch.txt')
        #trainset RGB-0 sketch-1
        self.selected_train = {}
        self.selected_train = defaultdict(list)

        rgb_data_file_list = open(train_visible_path, 'rt').read().splitlines()
        file_label = [int(s.split(' ')[1]) for s in rgb_data_file_list]
        # pid2label_img = {pid: label for label, pid in enumerate(np.unique(file_label))}
        # file_cam = [int(0) for s in rgb_data_file_list]
        self.person2text  = {pid: label for label, pid in enumerate(np.unique(file_label))}
        
        for j in range(len(rgb_data_file_list)):
            rgb_img_path = os.path.join(self.data_path, rgb_data_file_list[j].split(' ')[0])
            rgb_id = int(rgb_data_file_list[j].split(' ')[1])
            rgb_label = self.person2text[rgb_id]
            self.selected_train[rgb_label].append((rgb_img_path, rgb_label, 0))

        sketch_data_file_list = open(train_sketch_path, 'rt').read().splitlines()
        for j in range(len(sketch_data_file_list)):
            sketch_img_path = os.path.join(self.data_path, sketch_data_file_list[j].split(' ')[0])
            sketch_id = int(sketch_data_file_list[j].split(' ')[1])        
            sketch_label = self.person2text[sketch_id]
            self.selected_train[sketch_label].append((sketch_img_path, sketch_label, 1))

    def __len__(self):
        return len(self.selected_train)

    def __getitem__(self, index):
        rgb_img_path, rgb_label, rgb_trackid = self.selected_train[index][0]

        if len(self.selected_train[index][0:]):
            n = random.randint(1,len(self.selected_train[index][0:])-1)
            sketch_img_path, sketch_label, sketch_trackid = self.selected_train[index][n]
            assert sketch_trackid==1

        image = Image.open(rgb_img_path).convert('RGB')
        image = self.transform(image)

        sketch = Image.open(sketch_img_path).convert('RGB')
        sketch = self.transform(sketch)

        assert rgb_label==sketch_label

        label = rgb_label

        return {
            'image': image, 
            'sketch': sketch, 
            'id': label}


class QMUL_Shoe_img(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform

        test_visible_path = os.path.join(self.data_path, 'ShoeV2/test_visible.txt')

        rgb_gallery_file_list = open(test_visible_path, 'rt').read().splitlines()

        self.gallery_path = []
        self.img2person = [] 
        for j in range(len(rgb_gallery_file_list)):
            img_path = os.path.join(self.data_path, rgb_gallery_file_list[j].split(' ')[0])
            id = int(rgb_gallery_file_list[j].split(' ')[1])
            self.gallery_path.append(img_path)
            self.img2person.append(id)

        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.gallery_path)

    def __getitem__(self, index):
        rgb_img_path = self.gallery_path[index]
        label = self.img2person[index]

        image = Image.open(rgb_img_path).convert('RGB')
        image = self.transform(image)

        return image


class QMUL_Shoe_Test_query(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform

        test_sketch_path = os.path.join(self.data_path, 'ShoeV2/test_sketch.txt')

        sketch_query_file_list = open(test_sketch_path, 'rt').read().splitlines()

        self.sketch_path = []
        self.txt2person = []

        for j in range(len(sketch_query_file_list)):
            sketch_path = os.path.join(self.data_path, sketch_query_file_list[j].split(' ')[0])
            id = int(sketch_query_file_list[j].split(' ')[1])
            self.sketch_path.append(sketch_path)
            self.txt2person.append(id)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)

        text_file = os.path.join(self.data_path, 'ShoeV2/shoe_finetune_testB.json')
        text_anns = json.load(open(text_file))
        self.captions_dict = defaultdict(list)
        for obj in text_anns:
            id_index = int(obj['img_path'].split('.')[0])
            self.captions_dict[id_index].append(obj['caption'])     #{id1:[caption1, caption2], id2:[caption1]}

    def __len__(self):
        return len(self.sketch_path)

    def __getitem__(self, index):
        sketch_img_path = self.sketch_path[index]
        label = self.txt2person[index]

        sketch = Image.open(sketch_img_path).convert('RGB')
        sketch = self.transform(sketch)

        captions_list = self.captions_dict[label.item()]
        text = random.choice(captions_list)

        text_with_blank = 'a photo of * , and {}'.format(text)
        
        return text, sketch, text_with_blank
