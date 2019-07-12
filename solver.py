from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sys import exit
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import random



class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, content_loader, style_loaders, args):
        """Initialize argsurations."""

        # Data loader.
        self.content_loader = content_loader
        self.style_loaders = style_loaders
        self.style_iters = [iter(style_loader) for style_loader in style_loaders]
        self.args = args

        # Model argsurations.
        self.image_size = args.image_size
        self.c_dim = args.c_dim
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_fm = args.lambda_fm
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp
        self.reg_type = args.reg_type
        self.BCELoss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # Training argsurations.
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.result_dir = args.result_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step

        # Build the model and tensorboard.
        self.build_model()
        self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim)
        self.D = Discriminator(self.d_conv_dim, self.c_dim)
        self.generator = Generator(self.g_conv_dim).train(False)

        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        # For Adam (Unofficial)
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # For RMSprop(Official)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=0.0001)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=0.0001)

        self.accumulate(self.generator, self.G.module, 0)
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.generator.to(self.device)

        # weight init
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        self.generator.apply(self.weights_init)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        # elif classname.find('Norm') != -1:
        #     nn.init.normal_(m.weight.data, 1.0, 0.02)
        #     nn.init.constant_(m.bias.data, 0)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def restore_model(self, resume_iters):
        """Restore the trained generator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard writer."""
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, x_real, x_fake):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        y_fake_hat, _ = self.D(x_hat)

        y = y_fake_hat
        x = x_hat

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg


    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def choose_label(self):
        A = random.randint(0, self.args.c_dim-1)
        B = random.randint(0, self.args.c_dim-1)
        if A != B:
            return A, B
        else:
            return self.choose_label()


    def compute_triplet_loss(self, anchor=None, negative=None, positive=None):
        A = - torch.abs(anchor - positive)
        denominator = A.exponential_(1).mean()
        B = - torch.abs(anchor - negative)
        numerator = (B.exponential_(1) + A.exponential_(1)).mean()

        loss = (- torch.log(denominator / numerator))#.mean()

    
        return loss

    def train(self):
        """Train StarGAN within a single dataset."""
        print('Start training...')
        loss = {}
        counter = 0
        for epoch in range(self.args.num_epochs):
            self.G.train()
            self.generator.train()
            
            p_bar = tqdm(self.content_loader)
            for j, (x_real, _)in enumerate(p_bar):
                
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                x_real =  x_real.to(self.device).requires_grad_()         # Input contents.

                x_styles = []         # Input styles. During training, 
                # G learns to translate images between two randomly sampled source classes.
                # In the implementation section, I can not understand `we train the FUNIT model using K = 1`.
                A, B = self.choose_label()
                for i in [A, B]:
                    try:
                        x_style, _ = next(self.style_iters[i])
                    except:
                        self.style_iters[i] = iter(self.style_loaders[i])
                        x_style, _ = next(self.style_iters[i])
                    x_styles.append(x_style)

                    # TODO: DEBUG whether can sample all classes?
                #     save_image(self.denorm(x_style), f"{self.args.sample_dir}/x_style_{i}.png")
                # save_image(self.denorm(x_real), f"{self.args.sample_dir}/x_real.png")
                # exit()

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                
                self.reset_grad()
                
                # Compute loss with real images.
                y_real, feature_real = self.D(x_real)

                if self.args.loss_type == "wgangp":
                    d_loss_real = - y_real.mean()
                elif self.args.loss_type == "hinge":
                    d_loss_real = nn.ReLU()(1.0 - y_real).mean()
                    # if self.reg_type == 'real' or self.reg_type == 'real_fake':
                    #     d_loss_real.backward(retain_graph=True)
                    #     reg = self.lambda_gp * self.compute_grad2(y_real, x_real).mean()
                    #     reg.backward()
                    #     loss['D/loss_reg_real'] = reg.item()
                    # else:
                    #     d_loss_real.backward()
                elif self.args.loss_type == "bce":
                    label = torch.full((self.args.batch_size * self.args.c_dim,), 1).to(self.device)
                    d_loss_real = self.BCELoss(self.sigmoid(y_real.view(-1)), label)
                elif self.args.loss_type == "ls":
                    label = torch.full((self.args.batch_size * self.args.c_dim,), 1).to(self.device)
                    d_loss_real = 0.5 * torch.mean((y_real.view(-1) - label) ** 2)
                d_loss_real.backward(retain_graph=True)


                # Compute loss with fake images.
                x_fake = self.G(x_real, x_styles)
                y_fake, feature_fake = self.D(x_fake.detach())

                # Feature matching loss.
                _, feature_y1 = self.D(x_styles[0].cuda())
                _, feature_y2 = self.D(x_styles[1].cuda())
                mean_feature = (feature_y1 + feature_y2) / 2
                # d_loss_fm = torch.abs(y_feature - (feature_y1 + feature_y2)/2).mean()

                d_loss_fm = self.compute_triplet_loss(
                    anchor=feature_fake, negative=feature_real, positive=mean_feature).mean() * self.lambda_fm
                d_loss_fm.backward(retain_graph=True)

                if self.args.loss_type == "wgangp":
                    d_loss_fake = y_fake.mean()
                    d_loss_fake.backward()

                    d_loss_gp = self.gradient_penalty(x_real, x_fake) * self.lambda_gp
                    d_loss_gp.backward()
                    loss['D/loss_gp'] = d_loss_gp.item()
                elif self.args.loss_type == "hinge":
                    d_loss_fake = nn.ReLU()(1.0 + y_fake).mean()
                    d_loss_fake.backward()

                    # if self.reg_type == 'fake' or self.reg_type == 'real_fake':
                    #     d_loss_fake.backward(retain_graph=True)
                    #     reg = self.lambda_gp * self.compute_grad2(y_fake, x_fake).mean()
                    #     reg.backward()
                    #     loss['D/loss_reg_fake'] = reg.item()
                    # else:
                        # d_loss_fake.backward()
                
                elif self.args.loss_type == "bce":
                    label.fill_(0)
                    d_loss_fake = self.BCELoss(self.sigmoid(y_fake.view(-1)), label)
                    d_loss_fake.backward()

                elif self.args.loss_type == "ls":
                    label.fill_(0)
                    d_loss_fake = 0.5 * torch.mean((y_fake.view(-1) - label) ** 2)
                    d_loss_fake.backward()
                
                self.d_optimizer.step()

                # Logging.
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_fm'] = d_loss_fm.item()
                
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                
                if (j+1) % self.n_critic == 0:
                    # Adv Loss
                    x_fake = self.G(x_real, x_styles)
                    y_fake, _ = self.D(x_fake)

                    if self.args.loss_type == "bce":
                        label.fill_(1)
                        g_loss_fake = self.BCELoss(self.sigmoid(y_fake.view(-1)), label)
                    elif self.args.loss_type == "ls":
                        label.fill_(1)
                        g_loss_fake = 0.5 * torch.mean((y_fake.view(-1) - label)**2)
                    else:
                        g_loss_fake = - y_fake.mean()

                    # Rec Loss
                    g_loss_rec = (torch.abs(x_real - x_fake)).mean()

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    self.accumulate(self.generator, self.G.module)

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()

                # Print out training information.
                log = f"Epoch[{epoch}/{self.args.num_epochs}]"
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                    self.writer.add_scalar(tag, value, counter)
                p_bar.set_description(log)
                counter += 1

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Translate fixed images for debugging.
            if (epoch+1) % self.sample_step == 0:
                self.G.eval()
                self.generator.eval()
                with torch.no_grad():
                    x_concat = torch.cat([x_real, x_styles[0].to(self.device), x_styles[1].to(self.device), x_fake], dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-G.png'.format(epoch))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                with torch.no_grad():
                    x_fake = self.generator(x_real, x_styles)
                    x_concat = torch.cat([x_real, x_styles[0].to(self.device), x_styles[1].to(self.device), x_fake], dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-accum-G.png'.format(epoch))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

            # Save model checkpoints.
            if (epoch+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(epoch))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(epoch))
                G_accum_path = os.path.join(self.model_save_dir, '{}-accm-G.ckpt'.format(epoch))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.generator.state_dict(), G_accum_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):
        os.makedirs(self.args.generated_dir , exist_ok=True)

        self.restore_model(self.args.test_model_epoch)
        
        content_iter = iter(self.content_loader)
        
        print("Start generate fake images")
        for dim in range(self.args.c_dim):
            for j in tqdm(range(self.args.num_output_each_dim)):
                x_real, _ = next(content_iter)
                x_real = x_real.to(self.device)

                x_styles = []
                for _ in range(self.args.num_input_styles):
                    x_style, _ = next(self.style_iters[dim])
                    x_styles.append(x_style)
                
                with torch.no_grad():
                    x_fake = self.G(x_real, x_styles)
                
                x_concat = x_real
                for x_style in x_styles:
                    x_concat = torch.cat([x_concat, x_style.to(self.device)], dim=3)
                x_concat = torch.cat([x_concat, x_fake], dim=3)

                sample_path = os.path.join(self.args.generated_dir, f'{dim}-{j}-{self.args.test_model_epoch}.png')
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            print(f"Finish {dim}")