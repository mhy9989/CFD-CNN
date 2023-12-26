import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import print_log
import os
import os.path as osp

def plot_test_figure(x_site_matrix, y_site_matrix, min_max, data, data_select, data_name, mode, pic_folder, dpi=300):
    cmap = 'RdBu_r'
    levels = np.linspace(min_max[0], min_max[1], 600)
    map = plt.contourf(x_site_matrix, y_site_matrix, data, levels,cmap=cmap) 
    pic_name = f'{data_select}_{mode}_{data_name}.png'
    ax = plt.gca()
    ax.set_aspect(1) 
    plt.colorbar(map,fraction=0.02, pad=0.03,
                    ticks=np.linspace(min_max[0], min_max[1], 5),
                    format = '%.1e')
    plt.title(f"{mode} {data_name} data of type {data_select}")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    pic_path = osp.join(pic_folder, pic_name)
    plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
    print_log(f'{data_select}_{mode}_{data_name} picture is saved')
    plt.close()

def plot_learning_curve(loss_record, model_path, dpi=300, title='', dir_name = "pic"):
    ''' Plot learning curve of your DNN (train & valid loss) '''
    total_steps = len(loss_record['train_loss'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train_loss']) // len(loss_record['valid_loss'])]
    plt.semilogy(x_2, loss_record['valid_loss'], c='tab:cyan', label='valid')
    plt.semilogy(x_1, loss_record['train_loss'], c='tab:red', label='train')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()

    pic_name = f'loss_record.png'
    pic_folder = osp.join(model_path, dir_name)
    os.makedirs(pic_folder, exist_ok=True)
    pic_path =osp.join(pic_folder, pic_name)
    print_log(f'Simulation picture saved in {pic_path}')
    plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
    plt.close()
