import copy
import time

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from utils import criterion_multi_classes
from utils import molecules_custom_dataset


def train_model(model, criterion, optimizer, scheduler, num_epochs, tot_classes, train_loader, val_loader):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 100
    len_train_loader = len(train_loader)

    for epoch in range(num_epochs):

        for phase in ['train']:  # , 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = torch.Tensor(tf.one_hot(labels, tot_classes).numpy())
                labels = labels.cuda()

                task0 = labels[:, 0]
                task1 = labels[:, 1]
                task2 = labels[:, 2]
                task3 = labels[:, 3]
                task4 = labels[:, 4]
                task5 = labels[:, 5]
                task6 = labels[:, 6]
                task7 = labels[:, 7]
                task8 = labels[:, 8]
                task9 = labels[:, 9]
                task10 = labels[:, 10]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss0 = criterion[0](outputs[0], torch.max(task0.float(), 1)[1])
                    loss1 = criterion[1](outputs[1], torch.max(task1.float(), 1)[1])
                    loss2 = criterion[2](outputs[2], torch.max(task2.float(), 1)[1])
                    loss3 = criterion[3](outputs[3], torch.max(task3.float(), 1)[1])
                    loss4 = criterion[4](outputs[4], torch.max(task4.float(), 1)[1])
                    loss5 = criterion[5](outputs[5], torch.max(task5.float(), 1)[1])
                    loss6 = criterion[6](outputs[6], torch.max(task6.float(), 1)[1])
                    loss7 = criterion[7](outputs[7], torch.max(task7.float(), 1)[1])
                    loss8 = criterion[8](outputs[8], torch.max(task8.float(), 1)[1])
                    loss9 = criterion[9](outputs[9], torch.max(task9.float(), 1)[1])
                    loss10 = criterion[10](outputs[10], torch.max(task10.float(), 1)[1])

                    if phase == 'train':
                        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader)

            print(f"{phase} epoch {epoch} total loss: {loss:.4f}  task0 loss: {loss0:.3f} task1 loss: {loss1:.3f}")

            if phase == 'val' and epoch_loss < best_acc:
                print('saving with loss of {}'.format(epoch_loss), 'improved over previous {}'.format(best_acc))
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


class multi_tasking_model(torch.nn.Module):
    def __init__(self, tot_classes, heads_input_size=500):
        super(multi_tasking_model, self).__init__()

        self.tot_classes = tot_classes
        self.heads_input_size = heads_input_size

        self.conv_1 = nn.Conv1d(1, 32, kernel_size=2)
        self.conv_2 = nn.Conv1d(32, 64, kernel_size=2)
        self.conv_3 = nn.Conv1d(64, 20, kernel_size=2)
        self.x1 = nn.Linear(2400, 5000)
        self.x1_ = nn.Linear(5000, 2500)
        nn.init.xavier_normal_(self.x1.weight)
        self.x2 = nn.Linear(2500, 1000)
        self.x2_ = nn.Linear(1000, self.heads_input_size)
        nn.init.xavier_normal_(self.x2.weight)
        self.x3 = nn.Linear(self.heads_input_size, self.heads_input_size)
        nn.init.xavier_normal_(self.x3.weight)
        nn.init.xavier_normal_(self.conv_1.weight)
        nn.init.xavier_normal_(self.conv_2.weight)
        nn.init.xavier_normal_(self.conv_3.weight)

        # heads
        self.y1o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y2o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y3o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y4o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y5o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y6o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y7o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y8o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y9o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y10o = nn.Linear(self.heads_input_size, self.tot_classes)
        self.y11o = nn.Linear(self.heads_input_size, self.tot_classes)
        nn.init.xavier_normal_(self.y1o.weight)
        nn.init.xavier_normal_(self.y2o.weight)
        nn.init.xavier_normal_(self.y3o.weight)
        nn.init.xavier_normal_(self.y4o.weight)
        nn.init.xavier_normal_(self.y5o.weight)
        nn.init.xavier_normal_(self.y6o.weight)
        nn.init.xavier_normal_(self.y7o.weight)
        nn.init.xavier_normal_(self.y8o.weight)
        nn.init.xavier_normal_(self.y9o.weight)
        nn.init.xavier_normal_(self.y10o.weight)
        nn.init.xavier_normal_(self.y10o.weight)

    def forward(self, x):
        # shared network
        x = x.unsqueeze(1)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.x1(x))
        x = F.relu(self.x1_(x))
        x = F.relu(self.x2(x))
        x = F.relu(self.x2_(x))
        x = F.relu(self.x3(x))

        # heads
        task0 = F.softmax(self.y1o(x), dim=1)
        task1 = F.softmax(self.y2o(x), dim=1)
        task2 = F.softmax(self.y3o(x), dim=1)
        task3 = F.softmax(self.y4o(x), dim=1)
        task4 = F.softmax(self.y5o(x), dim=1)
        task5 = F.softmax(self.y6o(x), dim=1)
        task6 = F.softmax(self.y7o(x), dim=1)
        task7 = F.softmax(self.y8o(x), dim=1)
        task8 = F.softmax(self.y9o(x), dim=1)
        task9 = F.softmax(self.y10o(x), dim=1)
        task10 = F.softmax(self.y11o(x), dim=1)

        return task0, task1, task2, task3, task4, task5, task6, task7, task8, task9, task10


if __name__ == "__main__":
    device = "cuda:0"
    tot_classes = 2
    model = multi_tasking_model(tot_classes)
    model = model.to(device)
    print(model)
    print(model.parameters())
    criterion = criterion_multi_classes
    # criterion = criterion_binary_loss
    assert len(criterion) == 11, len(criterion)

    batch_size = 32
    lr = .001

    x, y = molecules_custom_dataset("./data/data_train_descriptors.csv")
    dataset = TensorDataset(x, y)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [10000, 1000, 1000])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=len(val_set))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lrmain, momentum=0.99)

    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    num_epochs = 30
    train_model(model, criterion, optimizer, lr_scheduler, num_epochs, tot_classes, train_loader, val_loader)
