import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from other import Other

voc_path = 'data/VOC2012'
voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
               'tv/monitor']

# RGB color for each class
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


class Datasets:
    def read_images(self):
        # 对图片进行初始化
        # transforms.Normalize 对数据按通道进行标准化，即先减均值，再除以标准差
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Resize((224, 224)),
             # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
             ])

        self.trainset = datasets.VOCSegmentation(root='data/', year='2012', image_set='train',
                                                 download=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=1, drop_last=True,
                                                       collate_fn=lambda x: x, shuffle=True, num_workers=0)

        return self.trainset, self.trainloader


if __name__ == '__main__':
    trainset, trainloader = Datasets().read_images()

    print(len(trainloader), type(trainloader), trainloader)

    for labels, data in enumerate(trainloader, 0):
        print(labels)
