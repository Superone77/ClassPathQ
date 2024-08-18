class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, model, device, trainloader, valloader, testloader, stat_loader, criterion,
                 instance_name,config_dict):

        self.model = model
        self.model.eval()
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.stat_loader = stat_loader
        self.base_criterion = criterion
        self.load_config(config_dict)
        self.instance_name = instance_name

        self.gradients = []
        self.activations = []
        
    
    def train(self, epoch, trainloader, optimizer, criterion):
        pass

    def load_config(self, config_dict):
        pass