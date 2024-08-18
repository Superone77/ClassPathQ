import torch

class SimpleTrainer(Trainer):

    def __init__(self, model, device, trainloader, valloader, testloader, stat_loader, criterion,
                 instance_name,config_dict):
        Trainer.__init__(self, model, device, trainloader, valloader, testloader, stat_loader, criterion,
                 instance_name,config_dict)

        self.load_config(config_dict)

    def train(self, epoch, trainloader, optimizer, criterion):
            print('Epoch: %d        instance: %s' % (epoch, self.instance_name))
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                self.model.train()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_batch = targets.size(0)
                correct_batch = predicted.eq(targets).sum().item()
                total += total_batch
                correct += correct_batch

            acc = 100. * correct / total
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss / (batch_idx + 1), acc, correct, total))
            return acc

    def test(self, testloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.base_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            loss_output = test_loss / (batch_idx + 1)
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss_output, acc, correct, total))

        return acc, loss
    
    def load_config(self, config_dict):
        pass
        