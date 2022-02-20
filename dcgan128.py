from __future__ import print_function
import os
import sys
from glob import glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

"""
Sample code for DCGAN, include the structure and train function.
You can start form this file or write your own structure and train function.
You can also modify anything in this file for training.

What you need to do:
1. Load your dataset
2. Train the DCGAN models and generate images in 3*3 grid.
3. Plot the generator and discriminator loss.
4. Interpolate the z vector and genrate 3*10 image.
5. Use the GAN for training and compare the results.

For more details, please reference to GANPractice.pdf.
Welcome to contact TA if you have any questions.
Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

# Set random seed for reproducibility
manualSeed = 2333
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 12
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100

# 160 40 # NSFW
# 100 10 # horoscope

# Size of feature maps in generator
ngf = 160
# Size of feature maps in discriminator
ndf = 40

# Number of training epochs
num_epochs = 1000
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                      else "cpu")
# G_loss D_loss list
G_losses = []
D_losses = []

# Plot Generator and Discriminator Loss
def loss_plot():
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="Generator")
    plt.plot(D_losses,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Loss_dcgan.png')
    #plt.show()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
    
    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)
print(netG)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netG.apply(weights_init)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
           # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )
    
    def forward(self, input):
        return self.main(input)
    

# Create the Discriminator
netD = Discriminator(ngpu).to(device)
print(netD)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netD.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fixed_noise_9 = torch.randn(9, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# SGD optimizers
#optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
#optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)


# Root directory for dataset

# Penis
# data_dir = '/media/poyao/D22C89212C890229/PenisDataset/raw_data'

# NSFW
# data_dir = '~/Downloads/GAN/DCGAN-tensorflow-master/data'

# NTHU HW2
data_dir = './portrait_data'

# horoscope
# data_dir = '/home/poyao/Downloads/poyao-2020/horoscope'

# kiss
# data_dir = '/home/poyao/Downloads/hw2/kissdata'

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

'''
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
'''

# Training Loop
def train_GAN():
    print("Starting Training Loop...")
    outputD_real = 0.0
    outputD_noise = 0.0
    outputG = 0.0
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            outputD_real = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(outputD_real, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            outputD_noise = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(outputD_noise, label)
            # Calculate the gradients for this batch
            errD_fake.backward()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of
            # all-fake batch through D
            outputG = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(outputG, label)
            # Calculate gradients for G
            errG.backward()

            # Update G
            optimizerG.step()
            
            # Save the images for checking generate result
            sample_images = fake

            # Store Loss
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            
            # training stats
            D_x = outputD_real.mean().item()
            D_G_z1 = outputD_noise.mean().item()
            D_G_z2 = outputG.mean().item()
            
            if i % 10 == 0:
                # Output training stats every epoch
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Image 
                # save_image(sample_images.data[:9], "images/%d.png" % epoch, nrow=3, normalize=True)
                
                # vutils.save_image(real_cpu,'images/real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),'images/fake_samples_epoch_%03d-%03d.png' % (epoch,i),normalize=True, padding=0)

        # Save DCGAN Model every epoch
        fname_gen_pt = 'models/dcgan-epoch-{}-gen.pt'.format(epoch + 1)
        # fname_disc_pt = 'models/dcgan-epoch-{}-disc.pt'.format(epoch + 1)
        torch.save(netG.state_dict(), fname_gen_pt)
        # torch.save(netD.state_dict(), fname_disc_pt)
        
       
def interpolation(lambda1, model, latent_1, latent_2):
    with torch.no_grad():

        # interpolation of the two latent vectors
        inter_latent = lambda1* latent_1 + (1- lambda1) * latent_2

        # reconstruct interpolated image
        inter_latent = inter_latent.to(device)
        inter_image = model(inter_latent)
        inter_image = inter_image.cpu()
        return inter_image

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def GenerateInterHuge(num):
    
    # Generate Interpolation Images
    for i, data in enumerate(dataloader, 0):
        print('Generating Interpolation dataloader',i,'/',len(dataloader))

        # sample two latent vectors from the standard normal distribution
        latent_1 = torch.randn(64, nz, 1, 1, device=device)
        if i == 0 :
            #print(':)')
            latent_2 = torch.randn(64, nz, 1, 1, device=device)
        #print(latent_1[0][0],latent_2[0][0])

        # interpolation lambdas
        lambda_range=np.linspace(0,1,num)

        for ind,l in enumerate(lambda_range):

            with torch.no_grad():
                # interpolation of the two latent vectors
                inter_latent = float(l)* latent_1 + (1- float(l)) * latent_2
                #print('inter_latent',inter_latent[0][0])
                inter_image = netG(inter_latent)
                #vutils.save_image(inter_image[0],'interhuge_nsfw_128x128_20200709/%05d-%05d.png' % (i, ind),normalize=True, padding=0)
        
        latent_2 = latent_1

def Sample(num):
    for i in range(0,num):
        #fixed_noise_527 = torch.randn(527, nz, 1, 1, device=device)
        fixed_noise_1 = torch.randn(100, nz, 1, 1, device=device)
        print('Saving samples/fake_samples_%02d' % (i))
        fake = netG(fixed_noise_1)
        for j in range(0,100):
            vutils.save_image(fake[j],'samples/fake_samples_%02d-%02d.png' % (i,j), normalize=True, padding=0)
            #vutils.save_image(fake.detach(),'images/test.png',normalize=True, padding=0, nrow=31)

def GenerateInterpolation():

     # Generate Interpolation Images
     for i, data in enumerate(dataloader, 0):
          print('Generating Interpolation dataloader',i,'/',len(dataloader))
          real_cpu = data[0].to(device)
          b_size = real_cpu.size(0)

          # sample two latent vectors from the standard normal distribution
          latent_1 = torch.randn(b_size, nz, 1, 1, device=device)
          latent_2 = torch.randn(b_size, nz, 1, 1, device=device)

          # interpolation lambdas
          lambda_range=np.linspace(0,1,10)

          for ind,l in enumerate(lambda_range):
              inter_image=interpolation(float(l), netG, latent_1, latent_2)
              inter_image = to_img(inter_image)
         
              if ind == 0:
                  con = np.transpose(inter_image[0,:,:,:], (1, 2, 0))
              else:
                  con = np.concatenate((con, np.transpose(inter_image[0,:,:,:], (1, 2, 0))), axis=1)

          if i % 3 == 0:
              con1 = con
          elif i % 3 == 1:
              con2 = con
          else:
              con12 = np.concatenate((con1, con2), axis=0)
              con123 = np.concatenate((con12, con), axis=0) 
              fig = plt.imshow(con123)
              plt.axis('off')
              fig.axes.get_xaxis().set_visible(False)
              fig.axes.get_yaxis().set_visible(False)
              print('savefig:inter/{}.png'.format(i))
              plt.savefig('inter/{}.png'.format(i), bbox_inches='tight', pad_inches = 0, dpi = (300))

def GenerateInterHugeVideos(bags, images, internum):

    for bag in range(16,bags):

        directory = "/home/poyao/Downloads/hw2/hugeinter/"+str(bag)
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print("existed")

        # Generate Interpolation Images
        for i in range(0,images):
            print('Generating Interpolation',bag,'/',i,'/',images)

            # sample two latent vectors from the standard normal distribution
            latent_1 = torch.randn(64, nz, 1, 1, device=device)
            if i == 0 :
                #print(':)')
                latent_2 = torch.randn(64, nz, 1, 1, device=device)
            #print(latent_1[0][0],latent_2[0][0])

            # interpolation lambdas
            lambda_range=np.linspace(0,1,internum)

            for ind,l in enumerate(lambda_range):

                with torch.no_grad():
                    # interpolation of the two latent vectors
                    inter_latent = float(l)* latent_1 + (1- float(l)) * latent_2
                    #print('inter_latent',inter_latent[0][0])
                    inter_image = netG(inter_latent)
                    vutils.save_image(inter_image[0],'hugeinter/'+str(bag)+'/%05d-%05d.png' % (i, ind),normalize=True, padding=0)
            
            latent_2 = latent_1

if __name__ == '__main__':
    # Train the model
    # train_GAN()
    # loss_plot()

    netG.load_state_dict(torch.load('models/dcgan-epoch-103-gen.pt'))
    print(netG)
    GenerateInterpolation()
    # Sample(10)
    # GenerateInterHuge(200)
    # GenerateInterHugeVideos(20, 50, 200)


