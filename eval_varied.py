import time
from datetime import datetime
from pathlib import Path
import xlwt

from src.dataset.data_loader import GMDataset, get_dataloader
from torch_geometric import utils as tg_util
from src.evaluation_metric_varied import *
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda
from src.utils.timer import Timer
from src.utils.config import cfg


def eval_model(model, dataloader, vis=False, verbose=False, xls_sheet=None):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes

    recalls = []
    precisions = []
    f1s = []
    pred_time = []
    objs = torch.zeros(len(classes), device=device)
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        recall_list = []
        precision_list = [] 
        f1_list = []
        pred_time_list = []
        obj_total_num = torch.zeros(1, device=device)

        for inputs in dataloader:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs,model.module.device)

            batch_num = inputs['batch_size']

            iter_num = iter_num + 1
            output_file = './output_images/it_{:d}_'.format(iter_num)
            with torch.set_grad_enabled(False):
                timer.tick()
                outputs = model(inputs)
                pred_time_list.append(torch.full((batch_num,), timer.toc() / batch_num))

            # Evaluate matching accuracy
            assert 'perm_mat' in outputs
            assert 'gt_perm_mat' in outputs

            recall = matching_recall(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'])
            recall_list.append(recall)
            precision = matching_precision(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'])
            precision_list.append(precision)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1[torch.isnan(f1)] = 0
            f1_list.append(f1)

            if 'aff_mat' in outputs:
                pred_obj_score = objective_score(outputs['perm_mat'], outputs['aff_mat'])
                gt_obj_score = objective_score(outputs['gt_perm_mat'], outputs['aff_mat'])
                objs[i] += torch.sum(pred_obj_score / gt_obj_score)
                obj_total_num += batch_num

        if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
            running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
            print('Class {} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
            running_since = time.time()

        recalls.append(torch.cat(recall_list))
        precisions.append(torch.cat(precision_list))
        f1s.append(torch.cat(f1_list))
        objs[i] = objs[i] / obj_total_num
        pred_time.append(torch.cat(pred_time_list))

        if verbose:
            print('Class {} {}'.format(cls, format_accuracy_metric(precisions[i], recalls[i], f1s[i])))
            print('Class {} norm obj score = {:.4f}'.format(cls, objs[i]))
            print('Class {} pred time = {}s'.format(cls, format_metric(pred_time[i])))
            if cfg.PROBLEM.TYPE == 'MGMC':
                print('Class {} cluster acc={}'.format(cls, format_metric(cluster_acc[i])))
                print('Class {} cluster purity={}'.format(cls, format_metric(cluster_purity[i])))
                print('Class {} cluster rand index={}'.format(cls, format_metric(cluster_ri[i])))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    if xls_sheet:
        for idx, cls in enumerate(classes):
            xls_sheet.write(0, idx+1, cls)
        xls_sheet.write(0, idx+2, 'mean')

    xls_row = 1

    # show result
    print('Matching accuracy')
    if xls_sheet:
        xls_sheet.write(xls_row, 0, 'precision')
        xls_sheet.write(xls_row+1, 0, 'recall')
        xls_sheet.write(xls_row+2, 0, 'f1')
    for idx, (cls, cls_p, cls_r, cls_f1) in enumerate(zip(classes, precisions, recalls, f1s)):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_r, cls_f1)))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+1, torch.mean(cls_p).item())
            xls_sheet.write(xls_row+1, idx+1, torch.mean(cls_r).item())
            xls_sheet.write(xls_row+2, idx+1, torch.mean(cls_f1).item())
    print('average accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(recalls), torch.cat(f1s))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(precisions)).item())
        xls_sheet.write(xls_row+1, idx+2, torch.mean(torch.cat(recalls)).item())
        xls_sheet.write(xls_row+2, idx+2, torch.mean(torch.cat(f1s)).item())
        xls_row += 3

    if not torch.any(torch.isnan(objs)):
        print('Normalized objective score')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'norm objscore')
        for idx, (cls, cls_obj) in enumerate(zip(classes, objs)):
            print('{} = {:.4f}'.format(cls, cls_obj))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, cls_obj.item())
        print('average objscore = {:.4f}'.format(torch.mean(objs)))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, torch.mean(objs).item())
            xls_row += 1

    print('Predict time')
    if xls_sheet: xls_sheet.write(xls_row, 0, 'time')
    for idx, (cls, cls_time) in enumerate(zip(classes, pred_time)):
        print('{} = {}'.format(cls, format_metric(cls_time)))
        if xls_sheet: xls_sheet.write(xls_row, idx + 1, torch.mean(cls_time).item())
    print('average time = {}'.format(format_metric(torch.cat(pred_time))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(pred_time)).item())
        xls_row += 1

    return torch.Tensor(list(map(torch.mean, recalls)))


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict
    from src.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              problem=cfg.PROBLEM.TYPE,
                              length=cfg.EVAL.SAMPLES,
                              cls=cfg.EVAL.CLASS,
                              obj_resize=cfg.PROBLEM.RESCALE)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))

        model_path = ''
        if cfg.EVAL.EPOCH is not None and cfg.EVAL.EPOCH > 0:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.EVAL.EPOCH))
        if len(cfg.PRETRAINED_PATH) > 0:
            model_path = cfg.PRETRAINED_PATH
        if len(model_path) > 0:
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path, strict=False)

        eval_model(
            model, dataloader,
            vis=False,
            verbose=True,
            xls_sheet=ws
        )
    wb.save(str(Path(cfg.OUTPUT_PATH) / ('eval_result_' + now_time + '.xls')))
