import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)


def split_r_c(fea, r, c):
    # fea.shape = [1, 256, 190, 150]
    f_rows = fea.chunk(r, 2)  # a tuple, shape = [r], f_rows[0].shape = [1, 256, 19, 150]
    r_c = []
    for i in range(r):
        r_c.append(f_rows[i].chunk(c, 3))  # size=[r,c], r_c[0,0].shape = [1, 256, 19, 30]

    for i in range(r):
        if i == 0:
            f_new = torch.cat(r_c[i], 1)
        else:
            f_new_t = torch.cat(r_c[i], 1)
            f_new = torch.cat((f_new, f_new_t), 1)
    # f_new.shape = [1, 12800, 19, 30]
    return f_new

def merge_r_c(fea, r, c):
    # fea.shape = [1, 50, 19, 30]
    f_new_s = fea.chunk(r * c, 1)
    for i in range(r):
        if i == 0:
            f_re = torch.cat([f_new_s[k] for k in range(i * c, i * c + c)], 3)
        else:
            f_re_t = torch.cat([f_new_s[k] for k in range(i * c, i * c + c)], 3)
            f_re = torch.cat((f_re, f_re_t), 2) # [1, 1, 190, 150]
    return f_re

""" Local discriminator """
class netD_pixel(nn.Module):
    def __init__(self,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()


    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

""" Midle discriminator """
class netD_mid(nn.Module):
    def __init__(self,context=False):
        super(netD_mid, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat#torch.cat((feat1,feat2),1)#F
        else:
          return x

""" Global discriminator """
class netD(nn.Module):
    def __init__(self,context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x
          
""" Image discriminator """
class ImageDA(nn.Module):
    def __init__(self,dim):
        super(ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)

    def forward(self,x):
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        return x

""" Instance discriminator """
class InstanceDA(nn.Module):
    def __init__(self):
        super(InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(2048, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)

    def forward(self,x):
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        #x=self.clssifer(x)
        return x






""" ... discriminator """
class netD_da(nn.Module):
    def __init__(self, feat_d):
        super(netD_da, self).__init__()
        self.fc1 = nn.Linear(feat_d,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x  #[256, 2]


""" ... discriminator """
class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

""" ... discriminator """
class netD_m_pixel(nn.Module):
    def __init__(self,r = 10, c = 5):
        super(netD_m_pixel, self).__init__()
        self.row = r
        self.col = c
        self.group = int(r * c)
        self.channels_in = int(256 * r * c)
        self.channels_mid = int(128 * r * c)
        self.channels_out = int(r * c)
        self.conv1 = nn.Conv2d(self.channels_in, self.channels_mid, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self.conv2 = nn.Conv2d(self.channels_mid, self.channels_mid, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self.conv3 = nn.Conv2d(self.channels_mid, self.channels_out, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self._init_weights()
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = split_r_c(x, self.row,self.col) # [1, 12800, 19, 30]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # [1, 50, 19, 30]
        x = merge_r_c(x, self.row,self.col) #[1, 1, 190, 150]
        return F.sigmoid(x)