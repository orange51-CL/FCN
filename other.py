import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


# 工具类
class Other(object):
    # 展示图像的函数
    def imshow(self, img):
        self.img1 = img / 2 + 0.5  # unnormalize
        self.npimg = self.img1.numpy()

        # plt.imshow()函数负责对图像进行处理，并显示其格式
        plt.imshow(np.transpose(self.npimg, (1, 2, 0)))
        # plt.show()则是将plt.imshow()处理后的函数显示出来
        plt.show()

    # 将tensor格式的图片可视化
    def tensorToimg(self, tensor):
        self.unloader = transforms.ToPILImage()
        self.image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        self.image = self.image.squeeze(0)  # remove the fake batch dimension
        self.image = self.unloader(self.image)
        plt.imshow(self.image)
        plt.show()

    # 该标量值张量内的单个正确预测数
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
