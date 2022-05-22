import torch 
import torch.nn as nn

class color_temp_tuning_net(nn.Module):
    def __init__(self, ncInput=5+5):
        super(color_temp_tuning_net, self).__init__()
        self.fc1 = nn.Conv2d(ncInput, 128, kernel_size=1, stride=1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.relu2 = nn.PReLU()
        self.fc3 = nn.Conv2d(256, 34*3, kernel_size=1, stride=1, bias=False)

    def polynomial_kernel_function_generation(self, rgb, order=4):
        polynomial_kernel_function = []
        r = rgb[:,0:1,...]
        g = rgb[:,1:2,...]
        b = rgb[:,2:3,...]
        polynomial_kernel_function.append(r)
        polynomial_kernel_function.append(g)
        polynomial_kernel_function.append(b)
        polynomial_kernel_function.append(r*r)
        polynomial_kernel_function.append(g*g)
        polynomial_kernel_function.append(b*b)
        polynomial_kernel_function.append(r*g)
        polynomial_kernel_function.append(g*b)
        polynomial_kernel_function.append(r*b)
        polynomial_kernel_function.append(r**3)
        polynomial_kernel_function.append(g**3)
        polynomial_kernel_function.append(b**3)
        polynomial_kernel_function.append(r*g*g)
        polynomial_kernel_function.append(g*b*b)
        polynomial_kernel_function.append(r*b*b)
        polynomial_kernel_function.append(g*r*r)
        polynomial_kernel_function.append(b*g*g)
        polynomial_kernel_function.append(b*r*r)
        polynomial_kernel_function.append(r*g*b)
        if order == 3:
            return torch.cat(polynomial_kernel_function,dim=1)
        polynomial_kernel_function.append(r**4)
        polynomial_kernel_function.append(g**4)
        polynomial_kernel_function.append(b**4)
        polynomial_kernel_function.append(r**3*g)
        polynomial_kernel_function.append(r**3*b)
        polynomial_kernel_function.append(g**3*r)
        polynomial_kernel_function.append(g**3*b)
        polynomial_kernel_function.append(b**3*r)
        polynomial_kernel_function.append(b**3*g)
        polynomial_kernel_function.append(r**2*g**2)
        polynomial_kernel_function.append(g**2*b**2)
        polynomial_kernel_function.append(r**2*b**2)
        polynomial_kernel_function.append(r**2*g*b)
        polynomial_kernel_function.append(g**2*r*b)
        polynomial_kernel_function.append(b**2*r*g)

        return torch.cat(polynomial_kernel_function,dim=1)

    def weight_estimation_net(self, source, target):
        B = source.shape[0]
        source_label = torch.zeros(B,5).to(source.device)
        source_index = source.view(-1,1)
        source_label.scatter_(dim=1, index=source_index, value=1)
        source_label = source_label.view(B,5,1,1)

        target_label = torch.zeros(B,5).to(target.device)
        target_index = target.view(-1,1)
        target_label.scatter_(dim=1, index=target_index, value=1)
        target_label = target_label.view(B,5,1,1)

        label = torch.cat([source_label,target_label], dim=1)

        feat = self.relu1(self.fc1(label))
        feat = self.relu2(self.fc2(feat))
        weight = self.fc3(feat)

        return weight.view(B,3,34)

    def forward(self, source, target, source_rgb):
        weight = self.weight_estimation_net(source, target)# b,3,34
        polynomial_kernel_feature = self.polynomial_kernel_function_generation(source_rgb) # b,34,h,w
        b,c,h,w = polynomial_kernel_feature.shape
        polynomial_kernel_feature = polynomial_kernel_feature.view(b,c,-1)
        target_rgb = torch.matmul(weight, polynomial_kernel_feature)
        target_rgb = target_rgb.view(b,3,h,w)

        return target_rgb

if __name__ == '__main__':
    data = torch.randn(1,3,512,512)
    source = torch.LongTensor([2])
    target = torch.LongTensor([1])
    net = color_temp_tuning_net()
    out = net(source, target, data)
    print(out.shape)

