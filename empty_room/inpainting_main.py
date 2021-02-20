import argparse
from model import PConvUNet
from loss import *
from train_vgg16 import train_vgg16
from inpainting_DataLoader import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

# root
parser.add_argument("--img_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/image', type=str)
parser.add_argument("--mask_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/mask', type=str)
parser.add_argument("--label_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/label', type=str)
parser.add_argument("--vgg16_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/model/vgg16.pth', type=str)

# VGG16
parser.add_argument("--vgg16_num_epoch", default=50, type=int)
parser.add_argument("--vgg16_batch_size", default=20, type=int)
parser.add_argument("--vgg16_lr", default=0.001, type=float)
parser.add_argument("--vgg16_weight_decay", default = 1e-2, type= int)
parser.add_argument("--vgg16_num_channels", default = 3, type= int)
parser.add_argument("--vgg16_num_classes", default = 2, type= int)

# PUnet
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_epoch", default=1000, type=int)
# parser.add_argument("--initial_lr", default=0.0002, type=float)
parser.add_argument("--finetune_lr", default=0.00005, type=float)
parser.add_argument("--weight_decay", default = 0, type= int)

parser.add_argument("--valid_coef", default=1.0, type=float)
parser.add_argument("--hole_coef", default=6.0, type=float)
parser.add_argument("--tv_coef", default=0.1, type=float)
parser.add_argument("--perc_coef", default=0.05, type=float)
parser.add_argument("--style_coef", default=120.0, type=float)
parser.add_argument("--tv_loss", default='mean', type=str)

# load model
parser.add_argument("--start_iter", default=0, type=int)

# save root
parser.add_argument("--mask_out_dir", default = '/home/siwoo/Desktop/kpmg_image/empty_room/mask', type= str)
parser.add_argument("--model_out_dir", default= '/home/siwoo/Desktop/kpmg_image/empty_room/model', type= str)

# pass
parser.add_argument("--train_vgg16", default= False, type=bool)
parser.add_argument("--make_mask", default= False, type=bool)
parser.add_argument("--train", default= True, type=bool)


args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.train_vgg16 == True:
    print('start training vgg16...')
    train_vgg16(args, device)
    print('done!')

if args.make_mask == True:
    print('make mask...')
    make_traindataset(args.img_root, args.mask_root, args.label_root)
    print('done!')


dataset = Inpainting_Dataset(args.img_root, args.mask_root, args.label_root)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset, batch_size=1)

# model = UNet(3,3)
model = PConvUNet(finetune=False, layer_size=7)
if args.start_iter != 0:
    model.load_state_dict(torch.load(os.path.join('/home/siwoo/Desktop/kpmg_image/empty_room/model/inpainting_'+str(args.start_iter)+'.pth')))
model.to(device)


criterion = InpaintingLoss(VGG16FeatureExtractor(args)).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                                    lr = args.finetune_lr, weight_decay = args.weight_decay)

if args.train == True:
    for epoch in range(args.num_epoch):
        total_loss = []
        for iter, (label, mask, img) in enumerate(dataloader):
            label = label.to(device)
            mask = mask.to(device)
            img = img.to(device)

            output, _ = model(label, mask)
            loss_dict = criterion(label, mask, output, img)
            loss = 0
            for key, val in loss_dict.items():
                coef = getattr(args, '{}_coef'.format(key))
                loss +=coef*val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss)

        print('epoch: {}/{} loss: {}'.format(epoch+1, args.num_epoch, sum(total_loss)/len(total_loss)))

        torch.save(model.state_dict(), '/home/siwoo/Desktop/kpmg_image/empty_room/model/inpainting_'+str(args.start_iter+args.num_epoch)+'.pth')


if args.train == False:
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(os.path.join('/home/siwoo/Desktop/kpmg_image/empty_room/model/inpainting_'+str(args.start_iter)+'.pth')))

        for iter, (label, mask, img) in enumerate(test_dataloader):
            label = label.to(device)
            mask = mask.to(device)
            img = img.to(device)

            output, _ = model(label, mask)
            mask = np.transpose(output[0].detach().cpu().numpy(), (1, 2, 0))
            mask = mask*255
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(args.mask_out_dir, str(iter)+'.png'))