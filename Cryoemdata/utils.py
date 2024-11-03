
import os
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def plot_confidence_interval_inf(confidence_data, labels_p, labels_t, result_dir, epoch='final', bin_num=10):
    labels_data = []
    accuracy_data = []
    precision_data = []
    interval_num_data = []
    interval_pos_num_data = []

    if labels_t is None:
        calculate_acc = False
    elif -1 in labels_t:
        calculate_acc = False
    else:
        calculate_acc = True
    for i in range(bin_num):
        confidence_min = i / bin_num
        confidence_max = (i + 1) / bin_num
        my_index = (confidence_data >= confidence_min) & (confidence_data < confidence_max)
        # confidence_data_i = confidence_data[my_index]
        labels_p_i = labels_p[my_index]

        interval_num_data.append(len(labels_p_i))
        pos_num = sum(labels_p_i == 1)
        interval_pos_num_data.append(pos_num)
        labels_data.append(str(confidence_min) + '-' + str(confidence_max))

        if calculate_acc and len(labels_p_i) > 0:
            labels_t_i = labels_t[my_index]
            accuracy_data_i = np.mean(labels_p_i == labels_t_i)
            accuracy_data.append(accuracy_data_i)
            precision_correct_num = sum(labels_p_i[labels_p_i == 1] == labels_t_i[labels_p_i == 1])
            if pos_num>0:
                precision_data.append((precision_correct_num / pos_num))
            else:
                precision_data.append(0)
        else:
            accuracy_data.append(0)
            precision_data.append(0)
    if not os.path.exists(result_dir + '/figures/'):
        os.makedirs(result_dir + '/figures/')
    my_width = 0.4
    x = np.arange(bin_num)
    fig_num, ax_num = plt.subplots()
    fig_num.set_size_inches((14, 8))

    num_bar = ax_num.bar(x - my_width / 2, interval_num_data, my_width,
                         label='particles number (all ' + str(len(labels_p)) + ')')
    ax_num.bar_label(num_bar, label_type='edge')

    num_bar_pos = ax_num.bar(x + my_width / 2, interval_pos_num_data, my_width,
                             label='positive particles number (all ' + str(sum(interval_pos_num_data)) + ')')
    ax_num.bar_label(num_bar_pos, label_type='edge')
    ax_num.set_xticks(x)
    ax_num.set_xticklabels(labels_data)
    ax_num.set_xlabel('confidence')
    ax_num.set_title('Particles num statistics')
    ax_num.legend()
    fig_num.savefig(result_dir + '/figures/confidence_interval_num_' + str(epoch) + '.png')

    if calculate_acc:
        fig_acc, ax_acc = plt.subplots()
        fig_acc.set_size_inches((14, 8))
        acc_bar = ax_acc.bar(x - my_width / 2, accuracy_data, my_width, label='classification accuracy')
        ax_acc.bar_label(acc_bar, labels=['%.3f' % acc if acc != 0 else str(0) for acc in accuracy_data],
                         label_type='edge')
        p_bar = ax_acc.bar(x + my_width / 2, precision_data, my_width, label='classification precision')
        ax_acc.bar_label(p_bar, labels=['%.3f' % p if p != 0 else str(0) for p in precision_data], label_type='edge')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels_data)
        ax_acc.set_xlabel('confidence')
        ax_acc.set_title('Classification performance')
        ax_acc.legend()
        fig_acc.savefig(result_dir + '/figures/confidence_interval_acc_' + str(epoch) + '.png')
    else:
        fig_acc = None
    return fig_num, fig_acc


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_clustering_labels(labels_path, epoch, clustering_labels):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path + '/')
    np.save(labels_path + '/predict_labels_epoch' + str(epoch) + '.npy', clustering_labels)


def save_acc_data(acc, acc_best, nmi, nmi_best, tb_writer, acc_sum, nmi_sum, clustering_times, epoch, out_path):
    if acc > acc_best:
        acc_best = acc
    if nmi > nmi_best:
        nmi_best = nmi
    acc_sum = acc_sum + acc
    nmi_sum = nmi + nmi_sum
    clustering_times = clustering_times + 1
    tb_writer.add_scalar("accuracy:", acc, epoch)
    tb_writer.add_scalar("nmi:", nmi, epoch)
    tb_writer.add_scalar("mean accuracy:", acc_sum / clustering_times,
                         clustering_times)
    tb_writer.add_scalar("mean NMI:", nmi_sum / clustering_times,
                         clustering_times)
    if not os.path.exists(out_path + 'acc_data/'):
        os.makedirs(out_path + 'acc_data/')
    with open(out_path + 'acc_data/' + 'acc_data.txt', 'w') as average_acc:
        average_acc.write('best accuracy' + str(acc_best))
        average_acc.write('\nbest NMI' + str(nmi_best))
        average_acc.write('\naverage accuracy' + str(acc_sum / clustering_times))
        average_acc.write('\naverage NMI' + str(nmi_sum / clustering_times))
    return acc_best, nmi_best, acc_sum, nmi_sum, clustering_times


def save_trained_model(model, optimizer, epoch, save_path):
    import torch
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': epoch + 1}, save_path + 'epoch' + str(epoch) + '_model.pth.tar')


def save_averages(average_imgs_all, average_generated_imgs=None, run_root_dir=None, epoch=0):
    import torch
    import mrcfile
    if not os.path.exists(run_root_dir + 'averages/'):
        os.makedirs(run_root_dir + 'averages/')
    save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
               run_root_dir + 'averages/clustering_result_' + str(epoch) + '.png')
    if average_generated_imgs is not None:
        save_image(torch.unsqueeze(torch.from_numpy(average_generated_imgs), 1),
                   run_root_dir + 'averages/generated_clustering_result_' + str(epoch) + '.png')
        # projectons_file = mrcfile.new(
        #     run_root_dir + 'averages/generated_clustering_averages_' + str(epoch) + '.mrcs',
        #     average_generated_imgs, overwrite=True)
    projectons_file = mrcfile.new(
        run_root_dir + 'averages/clustering_averages_' + str(epoch) + '.mrcs',
        average_imgs_all, overwrite=True)
    projectons_file.close()


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def free_mem():
    import os
    result = os.popen("fuser -v /dev/nvidia*").read()
    results = result.split()
    for pid in results:
        os.system(f"kill -9 {int(pid)}")


def numsCheng(i):
    for m in range(100000):
        m = i * 2
        pass
    return i * 2, 2


def multi_process_test():
    import torch
    import time
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from multiprocessing.pool import Pool

    print("start")
    time1 = time.time()
    nums_list = range(100000)
    pool = Pool(processes=10)
    result = pool.map(numsCheng, nums_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    print("end")
    # print(result)
    time2 = time.time()
    print("计算用时：", time2 - time1)

    # for i,j in result:
    #     print(i,j)



