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
###

class Datasets:
    def read_images(self):
        transform = transforms.Compose([transforms.ToTensor()])

        self.trainset = datasets.VOCSegmentation(root=voc_path, year='2012', image_set='train',
                                                 download=True, transform=transform)
        print(321)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
                                                       shuffle=True, num_workers=2)

        return self.trainset, self.trainloader


if __name__ == '__main__':
    trainset, trainloader = Datasets().read_images()
    dataiter = iter(trainloader)

    print(123)

    images, labels = dataiter.next()
    # 展示图像
    Other().imshow(torchvision.utils.make_grid(images))
    # 显示图像标签
    print(' '.join('%5s' % voc_classes[labels[j]] for j in range(4)))
