import os
import sys
import json
import numpy as np
import skimage.draw
import cv2
from xml.etree.ElementTree import parse
import torch
import torch.utils.data as data
import torch.nn.functional as F
from data.augmentations import *

def parse_xml(file, classes=['Cell']):
    ans = {}
    root = parse(file).getroot()
    objects = root.findall('object')
    ans['filename'] = file.split('/')[-1][:-4]
    ans['width'] = int(root.find('size').find('width').text)
    ans['height'] = int(root.find('size').find('height').text)
    ans['regions'] = []
    for item in objects:
        if item.find('name').text in classes:
            obj = {}
            obj['class'] = item.find('name').text
            points = []
            if item.find('polygon'):
                for i in item.find('polygon'):
                    points.append(int(i.text))
                obj['x'] = points[::2]
                obj['y'] = points[1::2]
                ans['regions'].append(obj)
    return ans

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks.append(torch.FloatTensor(sample[1][1]))
        num_crowds.append(sample[1][2])

    return imgs, (targets, masks, num_crowds)

class CellDataset(data.Dataset):
    def __init__(self, mode='train', root1='dataset1/', root2='dataset2/'):
        self.mode = mode
        self.root1 = root1
        self.root2 = root2
        name1 = root1 + mode + '.json'
        name2 = root2 + mode + '.json'
        with open(name1) as f:
            self.images1 = json.load(f)
        with open(name2) as f:
            self.images2 = json.load(f)
        self.annotations = []
        for name in self.images1:
            obj = parse_xml(root1 + 'annotations/'+name+'.xml')
            if len(obj['regions']) > 0:
                obj['root'] = 1
                self.annotations.append(obj)
        print("Dataset1 has "+str(len(self.annotations))+" samples!")
        for name in self.images2:
            obj = parse_xml(root2 + 'annotations/'+name+'.xml')
            if len(obj['regions']) > 0:
                obj['root'] = 2
                self.annotations.append(obj)
        print("Plus dataset2 has "+str(len(self.annotations))+" samples!")
        self.ids = range(len(self.annotations))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.annotations[image_id]
        #mask = np.zeros([info["height"], info["width"], len(info["regions"])], dtype=np.uint8)
        mask = np.zeros([len(info["regions"]), info["height"], info["width"]], dtype=np.uint8)
        for i, p in enumerate(info["regions"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['y'], p['x'])
            mask[i, rr, cc] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.float32), np.zeros([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.float32), np.zeros([mask.shape[0]], dtype=np.int32)

    def load_image(self, image_id):
        if self.annotations[image_id]['root'] == 1:
            image_path = self.annotations[image_id]['filename'] + '.jpg'
            image_path = os.path.join(self.root1, 'images', image_path)
        elif self.annotations[image_id]['root'] == 2:
            image_path = self.annotations[image_id]['filename'] + '.jpeg'
            image_path = os.path.join(self.root2, 'images', image_path)
        else:
            raise NotImplementedError
        try:
            img = cv2.imread(image_path)
            img = img[:,:,::-1]
        except:
            print(image_path)
        return np.array(img)

    def load_bbox(self, mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (x1, y1, x2, y2)].
        """
        boxes = np.zeros([mask.shape[0], 4], dtype=np.float32)
        h, w = mask.shape[1:]
        for i in range(mask.shape[0]):
            m = mask[i, :, :]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
                x1 /= w
                x2 /= w
                y1 /= h
                y2 /= h
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                # (x1, y1) left-up corner, (x2, y2) right-bottom corner
                x1, x2, y1, y2 = 0., 0., 0., 0.
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes

    def pull_item(self, image_id):
        img = self.load_image(image_id)
        masks, labels = self.load_mask(image_id)
        boxes = self.load_bbox(masks)
        #print(masks.sum())
        #print(boxes)
        if self.mode == 'train':
            img, masks, boxes, labels = self.transform_train(img, masks, boxes, labels)
        elif self.mode == 'val':
            img, masks, boxes, labels = self.transform_val(img, masks, boxes, labels)

        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, masks, 300, 300, 0

    def transform_train(self, img, masks, boxes, labels):
        composed_transforms = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            #Expand(),
            RandomSampleCrop(),
            RandomMirror(),
            RandomFlip(),
            Resize(300),
            ToPercentCoords(),
            Normalize(),
            ])
        return composed_transforms(img, masks, boxes, labels)


    def transform_val(self, img, masks, boxes, labels):
        composed_transforms = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            #PhotometricDistort(),
            #Expand(),
            #RandomSampleCrop(),
            #RandomMirror(),
            #RandomFlip(),
            Resize(300),
            ToPercentCoords(),
            Normalize(),
            ])
        return composed_transforms(img, masks, boxes, labels)
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train = CellDataset(mode='val', root1='../dataset1/', root2='../dataset2/')
    print(len(train))
    #dataloader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=detection_collate, num_workers=2, drop_last=True)
    #for ii, sample in enumerate(dataloader):
    #    print(len(sample))
    #    print(sample)
    def draw(img, masks):
        for i, m in enumerate(masks):
            i_ = i % 3
            h, w = m.shape
            mk = np.zeros([h, w, 3]).astype(np.float32)
            mk[:,:,i_] = m
            cv2.addWeighted(img, 1., mk, 1., 0, img)
        return img

    for i in range(20):
        sample = train.__getitem__(i)
        img = sample[0].numpy().transpose(1,2,0)
        gt, masks, _ = sample[1]
        #cv2.imshow('img', img[:,:,::-1])
        cv2.imshow('mask', draw(img, masks))
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
