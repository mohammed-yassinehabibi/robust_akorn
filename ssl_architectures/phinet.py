import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Dgrid(p, z, version='simplified',shift=512): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        z_shift = torch.roll(z,-shift,1)
        #return - F.cosine_similarity(p, z.detach(), dim=-1).mean() + F.cosine_similarity(p,z_shift.detach()).mean()
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean() + (F.cosine_similarity(p,z_shift.detach())**2).mean()
    else:
        raise Exception

def Dnsg(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        #z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            #Grid cell
            x = torch.sin(2*x) + x
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class decoder_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h's input and output (z and p) is d = 2048, 
        and h's hidden layer's dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return torch.tanh(x)

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h's input and output (z and p) is d = 2048, 
        and h's hidden layer's dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        #self.layer_common = nn.Linear(hidden_dim,512)

        #self.temperature = torch.arange(1,2049).view(-1,1).to('cuda:0')/2048*10.0


        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    #Grid cell like representation
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x


def ema_model(modelA, modelB, m):
    with torch.no_grad():
        for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
            paramA.data = m * paramA.data + (1 - m) * paramB.data
    return modelA

class XPhiNetTF(nn.Module):
    def __init__(self, backbone=resnet50(), mse_loss_ratio = 1, ori_loss_ratio=0, beta=0.99,out_dim=2048):
        super().__init__()
        
        self.backbone = backbone
        #self.projector = projection_MLP(backbone.fc.out_features,out_dim=out_dim)
        if isinstance(backbone, torchvision.models.resnet.ResNet):
            backbone_feature_dim = self.backbone.fc.out_features
        else:
            #self.projector = projection_MLP(in_dim=backbone.out[5].out_features,out_dim=out_dim)
            backbone_feature_dim = self.backbone.out.out_features
        self.projector = projection_MLP(in_dim=backbone_feature_dim,out_dim=out_dim)
        self.decoder = decoder_MLP(in_dim=out_dim, out_dim=out_dim)
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.slow_encoder = copy.deepcopy(self.encoder)
        self.predictor = prediction_MLP(in_dim=out_dim, out_dim=out_dim)
        self.mse_loss_ratio = mse_loss_ratio
        self.ori_loss_ratio = ori_loss_ratio
        self.beta = beta
        print('self.mse_loss_ratio = ', self.mse_loss_ratio)
        print('self.ori_loss_ratio = ', self.ori_loss_ratio)
    
    def forward(self, x1, x2,x_ori,sim2='mse', pred2='g', isSG=True):

        f, h, g = self.encoder, self.predictor, self.decoder
        z1, z2 = f(x1), f(x2)
        z_ori = self.slow_encoder(x_ori)
        p1, p2 = h(z1), h(z2)

        if pred2 == 'i':
           q1, q2 = p1, p2 
        elif pred2 == 'h':
           q1, q2 = h(p1), h(p2)
        elif pred2 == 'g':
           q1, q2 = g(p1), g(p2)

        mseloss = nn.MSELoss()
        L_sim = D(p1, z2) / 2 + D(p2,z1)/2
        
        #L_sim = mseloss(p1,z2.detach())/2 + mseloss(p2,z1.detach())/2
        if sim2 == 'mse':
           if isSG:
              L_mse = mseloss(q1, z_ori.detach()) / 2 + mseloss(q2,z_ori.detach())/2
           else:
              L_mse = mseloss(q1, z_ori) / 2 + mseloss(q2,z_ori)/2
        elif sim2 == 'cos':
           if isSG:
              L_mse = Dnsg(q1, z_ori.detach()) / 2 + Dnsg(q2,z_ori.detach())/2
           else:
              L_mse = Dnsg(q1, z_ori) / 2 + Dnsg(q2,z_ori)/2

        L = L_sim + L_mse
        self.slow_encoder = ema_model(self.slow_encoder, self.encoder, self.beta)
        return {'loss': L, 'loss_cos': L_sim, 'loss_mse': L_mse}