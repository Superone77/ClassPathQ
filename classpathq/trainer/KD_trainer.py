import torch
import datetime

class KDTrainer(SimpleTrainer):

    def __init__(self, model, device, trainloader, valloader, testloader, stat_loader, criterion,
                 instance_name,config_dict):
        Trainer.__init__(self, model, device, trainloader, valloader, testloader, stat_loader, criterion,
                 instance_name,config_dict)


    
    def init(self,KD_path):
        path =KD_path
        self.KLloss = torch.nn.KLDivLoss()
        self.model.load_state_dict(torch.load(path))

    def train(self, epoch, trainloader, optimizer, criterion):
        time_start = datetime.datetime.now()
        print('Epoch: %d        instance: %s' % (epoch, self.instance_name))
        self.model.eval()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs_KD = self.model_KD(inputs).detach().clone()
            loss = self.KD_loss_alpha * criterion(outputs, targets) + (1 - self.KD_loss_alpha) * self.KLloss(
                F.log_softmax(outputs, dim=1), F.softmax(outputs_KD, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_batch = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            total += total_batch
            correct += correct_batch

        acc = 100. * correct / total
        time_end = datetime.datetime.now()
        time_total = (time_end - time_start).seconds
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d), time: %ds'
              % (train_loss / (batch_idx + 1), acc, correct, total, time_total))