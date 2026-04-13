import argparse
from os.path import join
import torch
import pickle
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
#from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.visualizer as vis
import os
from torch import nn, optim
from qm9.property_prediction import prop_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
loss_l1 = nn.L1Loss()
def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    #assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, nodes_dist, prop_dist = get_latent_diffusion(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=64, iterations=20):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask, _ = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        prop_key1 = self.prop_dist.properties[0]
        prop_key2 = self.prop_dist.properties[1]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key1]['mean']
        else:
            context[0][0] = context[0][0] * self.prop_dist.normalizer[prop_key1]['mad'] + \
                            self.prop_dist.normalizer[prop_key1]['mean']
            context[0][1] = context[0][1] * self.prop_dist.normalizer[prop_key2]['mad'] + \
                         self.prop_dist.normalizer[prop_key2]['mean']
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            prop_key1: context[0][0].detach(),
            prop_key2: context[0][1].detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations




def test(model, epoch, loader, mean, mad, property, device, partition='train', optimizer=None, lr_scheduler=None,
          log_interval=20, debug_break=False):
    if partition == 'train':
        lr_scheduler.step()
    res = {'loss1': 0,'loss2': 0, 'counter': 0, 'loss1_arr': [],'loss2_arr': []}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model[0].eval()
            model[1].eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)
        # charges = data['charges'].to(device, dtype).squeeze(2)
        # nodes = prop_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label1 = data[property[0]].to(device, torch.float32)
        label2 = data[property[1]].to(device, torch.float32)


        pred1 = model[0](h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                     edge_mask=edge_mask,
                     n_nodes=n_nodes)

        pred2 = model[1](h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                      edge_mask=edge_mask,
                      n_nodes=n_nodes)


        loss1 = loss_l1(mad[0] * pred1 + mean[0], label1)
        loss2 = loss_l1(mad[1] * pred2 + mean[1], label2)

        res['loss1'] += loss1.item() * batch_size
        res['loss2'] += loss2.item() * batch_size
        res['counter'] += batch_size
        res['loss1_arr'].append(loss1.item())
        res['loss2_arr'].append(loss2.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss1 %.4f" % (
            epoch, i, sum(res['loss1_arr'][-10:]) / len(res['loss1_arr'][-10:])))
            print(prefix + "Epoch %d \t Iteration %d \t loss2 %.4f" % (
                epoch, i, sum(res['loss2_arr'][-10:]) / len(res['loss2_arr'][-10:])))
        if debug_break:
            break
    return res['loss1'] / res['counter'], res['loss2'] / res['counter']
def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class1_dir = args.classifier1_path
    classifier1 = get_classifier(class1_dir).to(args.device)

    class2_dir = args.classifier2_path
    classifier2 = get_classifier(class2_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, _ = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)
    classifier = [classifier1,classifier2]

    mean1, mad1 = property_norms[args.property1]['mean'], property_norms[args.property1]['mad']
    mean2, mad2 = property_norms[args.property2]['mean'], property_norms[args.property2]['mad']
    mean = [mean1,mean2]
    mad = [mad1,mad2]
    property = [args.property1,args.property2]

    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                               args.device, batch_size=args.batch_size, iterations=args.iterations)
    print(property,": We evaluate the classifier on our generated samples")
    loss = test(classifier, 0, diffusion_dataloader, mean, mad, property, args.device, partition='test', log_interval=1, debug_break=args.debug_break)

    print("Loss classifier on qm9_second_half: %.4f, %.4f" % loss)



def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/analysis/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x


def main_qualitative(args):
    args_gen = get_args_gen(args.generators_path)
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)

    for i in range(args.n_sweeps):
        print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
        save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, epoch=i, id_from=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_lumo_gap/')
    parser.add_argument('--classifier1_path', type=str, default='qm9/property_prediction/outputs/lumo')
    parser.add_argument('--classifier2_path', type=str, default='qm9/property_prediction/outputs/gap')
    parser.add_argument('--property1', type=str, default='lumo',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--property2', type=str, default='gap',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=256,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qm9_second_half',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)
