import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        #transforms.Resize(size=(512, 512)),
        #transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--content_seg_dir', type=str, required=True,
                    help='Directory path to a batch of content seg images')
parser.add_argument('--style_seg_dir', type=str, required=True,
                    help='Directory path to a batch of style seg images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
#parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--style_weight', type=float, default=500.0)
#parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=100.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)



cont_seg = Image.open(content_seg_path)
styl_seg = Image.open(style_seg_path)
cont_seg = np.asarray(cont_seg)
styl_seg = np.asarray(styl_seg)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images, content_path = next(content_iter)
    content_images = content_images.to(device)
    #content_images = next(content_iter).to(device)
    style_images, style_path = next(style_iter)
    style_images = style_images.to(device)
    content_seg_path = content_path.replace("content", "content_seg")
    content_seg_path = content_seg_path.replace("jpg", "png")
    style_seg_path = style_path.replace(".jpg", "_color.png")
    style_seg_path = style_seg_path.replace("style", "style_seg")
    loss_c, loss_s = network(content_images, style_images, cont_seg, styl_seg, device)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i+1) % 1000 == 0:
        print('iteration: ' + str(i) + '\n')
        print('loss_content: ' + str(loss_c.item()) + '\n')
        print('loss_style: ' + str(loss_s.item()) + '\n'))
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth.tar'.format(args.save_dir,
                                                           i + 1))
writer.close()
