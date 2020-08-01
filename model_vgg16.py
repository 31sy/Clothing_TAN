import torch.nn as nn
import torch.nn.functional as F
import models
import torch
import pdb
class TaskAttention(nn.Module):
    def __init__(self, num_tasks):
        super(TaskAttention, self).__init__()
        self.num_tasks = num_tasks

        self.attention_module = nn.Sequential(
             nn.Conv2d(512, 512, kernel_size=1, bias=True),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, self.num_tasks, kernel_size=1, bias=True),
             nn.Softmax(dim=1)
        )
      
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        att_value = self.attention_module(x)
        a_t = torch.split(att_value,1, dim=1)
        #pdb.set_trace()
        att_features = []
        for i in range(self.num_tasks):
            a_t_repeat = a_t[i].repeat(1,512,1,1) #expand a_t value to the same dimension with x
            att_feature = a_t_repeat * x
            att_feature = att_feature.view(x.size(0),512 * 7 * 7)
            att_features.append(self.classifier(att_feature))
        return att_features





class ClothingAttributeNet(nn.Module):
    def __init__(self, name, num_classes):
        super(ClothingAttributeNet, self).__init__()
        self.num_classes = num_classes

        if name == 'alexnet':
            self.backbone = models.alexnet(True)
        elif name == 'vgg16':
            self.backbone = models.vgg16(True)        

        self.task_attention =  TaskAttention(len(self.num_classes))
        self.attribute_feature = []
        self.attribute_feature = nn.ModuleList([self._make_feature(1024) for classes in num_classes])

        self.attribute_classifier = []
        self.attribute_classifier = nn.ModuleList([self._make_classifier(1024,classes) for classes in num_classes])

    def _make_feature(self, fc_size):
        fc_feature = nn.Linear(4096, fc_size)
        fc_relu = nn.ReLU(inplace=True)
        fc_drop = nn.Dropout()

        return nn.Sequential(fc_feature,fc_relu,fc_drop)

    def _make_classifier(self, fc_size, classes):
        output = nn.Linear(fc_size, classes)
        return nn.Sequential(output)


    def forward(self, x):
        x = self.backbone(x)
        x = self.task_attention(x)

        idx = 0
        fc = []
        for att_fc in self.attribute_feature:
            fc.append(att_fc(x[idx]))
            idx += 1
        idx = 0
        output = []
        for att_classifier in self.attribute_classifier:
            output.append(att_classifier(fc[idx]))
            idx += 1

        return output


