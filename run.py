import argparse
import os
import torch
import utils.evaluator as eu
from quicknat import QuickNat
from settings import Settings
from solver import Solver
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
import logging
import shutil

torch.set_default_tensor_type('torch.FloatTensor')


def load_data(data_params):
    print("Loading dataset")
    train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data


def train(train_params, common_params, data_params, net_params):
    train_data, test_data = load_data(data_params)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=train_params['val_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        quicknat_model = torch.load(train_params['pre_trained_path'])
    else:
        quicknat_model = QuickNat(net_params)

    solver = Solver(quicknat_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "betas": train_params['optim_betas'],
                                "eps": train_params['optim_eps'],
                                "weight_decay": train_params['optim_weight_decay']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'])

    solver.train(train_loader, val_loader)
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    quicknat_model.save(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']
    data_id = eval_params['data_id']

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    avg_dice_score, class_dist = eu.evaluate_dice_score(eval_model_path,
                                                        num_classes,
                                                        data_dir,
                                                        label_dir,
                                                        volumes_txt_file,
                                                        remap_config,
                                                        orientation,
                                                        prediction_path,
                                                        data_id,
                                                        device,
                                                        logWriter)
    logWriter.close()


def evaluate_bulk(eval_bulk):
    data_dir = eval_bulk['data_dir']
    prediction_path = eval_bulk['save_predictions_dir']
    volumes_txt_file = eval_bulk['volumes_txt_file']
    device = eval_bulk['device']
    #BORIS
    # label_names = ["vol_ID", "Background", "Left WM", "Left Cortex", "Left Lateral ventricle", "Left Inf LatVentricle",
    #                "Left Cerebellum WM", "Left Cerebellum Cortex", "Left Thalamus", "Left Caudate", "Left Putamen",
    #                "Left Pallidum", "3rd Ventricle", "4th Ventricle", "Brain Stem", "Left Hippocampus", "Left Amygdala",
    #                "CSF (Cranial)", "Left Accumbens", "Left Ventral DC", "Right WM", "Right Cortex",
    #                "Right Lateral Ventricle", "Right Inf LatVentricle", "Right Cerebellum WM",
    #                "Right Cerebellum Cortex", "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
    #                "Right Hippocampus", "Right Amygdala", "Right Accumbens", "Right Ventral DC"]
    label_names = ["background", "3rd-Ventricle", "4th-Ventricle", "Right-Accumbens-Area", "Left-Accumbens-Area", "Right-Amygdala", "Left-Amygdala", "Brain-Stem", "Right-Caudate", "Left-Caudate", "Right-Cerebellum-Exterior", "Left-Cerebellum-Exterior", "Right-Cerebellum-White-Matter", "Left-Cerebellum-White-Matter", "Right-Cerebral-White-Matter", "Left-Cerebral-White-Matter", "Right-Hippocampus", "Left-Hippocampus", "Right-Inf-Lat-Vent", "Left-Inf-Lat-Vent", "Right-Lateral-Ventricle", "Left-Lateral-Ventricle", "Right-Pallidum", "Left-Pallidum", "Right-Putamen", "Left-Putamen", "Right-Thalamus-Proper", "Left-Thalamus-Proper", "Right-Ventral-DC", "Left-Ventral-DC", "Cerebellar-Vermal-Lobules-I-V", "Cerebellar-Vermal-Lobules-VI-VII", "Cerebellar-Vermal-Lobules-VIII-X", "Left-Basal-Forebrain", "Right-Basal-Forebrain", "Right-ACgG--anterior-cingulate-gyrus", "Left-ACgG--anterior-cingulate-gyrus", "Right-AIns--anterior-insula", "Left-AIns--anterior-insula", "Right-AOrG--anterior-orbital-gyrus", "Left-AOrG--anterior-orbital-gyrus", "Right-AnG---angular-gyrus", "Left-AnG---angular-gyrus", "Right-Calc--calcarine-cortex", "Left-Calc--calcarine-cortex", "Right-CO----central-operculum", "Left-CO----central-operculum", "Right-Cun---cuneus", "Left-Cun---cuneus", "Right-Ent---entorhinal-area", "Left-Ent---entorhinal-area", "Right-FO----frontal-operculum", "Left-FO----frontal-operculum", "Right-FRP---frontal-pole", "Left-FRP---frontal-pole", "Right-FuG---fusiform-gyrus", "Left-FuG---fusiform-gyrus", "Right-GRe---gyrus-rectus", "Left-GRe---gyrus-rectus", "Right-IOG---inferior-occipital-gyrus", "Left-IOG---inferior-occipital-gyrus", "Right-ITG---inferior-temporal-gyrus", "Left-ITG---inferior-temporal-gyrus", "Right-LiG---lingual-gyrus", "Left-LiG---lingual-gyrus", "Right-LOrG--lateral-orbital-gyrus", "Left-LOrG--lateral-orbital-gyrus", "Right-MCgG--middle-cingulate-gyrus", "Left-MCgG--middle-cingulate-gyrus", "Right-MFC---medial-frontal-cortex", "Left-MFC---medial-frontal-cortex", "Right-MFG---middle-frontal-gyrus", "Left-MFG---middle-frontal-gyrus", "Right-MOG---middle-occipital-gyrus", "Left-MOG---middle-occipital-gyrus", "Right-MOrG--medial-orbital-gyrus", "Left-MOrG--medial-orbital-gyrus", "Right-MPoG--postcentral-gyrus", "Left-MPoG--postcentral-gyrus", "Right-MPrG--precentral-gyrus", "Left-MPrG--precentral-gyrus", "Right-MSFG--superior-frontal-gyrus", "Left-MSFG--superior-frontal-gyrus", "Right-MTG---middle-temporal-gyrus", "Left-MTG---middle-temporal-gyrus", "Right-OCP---occipital-pole", "Left-OCP---occipital-pole", "Right-OFuG--occipital-fusiform-gyrus", "Left-OFuG--occipital-fusiform-gyrus", "Right-OpIFG-opercular-part-of-the-IFG", "Left-OpIFG-opercular-part-of-the-IFG", "Right-OrIFG-orbital-part-of-the-IFG", "Left-OrIFG-orbital-part-of-the-IFG", "Right-PCgG--posterior-cingulate-gyrus", "Left-PCgG--posterior-cingulate-gyrus", "Right-PCu---precuneus", "Left-PCu---precuneus", "Right-PHG---parahippocampal-gyrus", "Left-PHG---parahippocampal-gyrus", "Right-PIns--posterior-insula", "Left-PIns--posterior-insula", "Right-PO----parietal-operculum", "Left-PO----parietal-operculum", "Right-PoG---postcentral-gyrus", "Left-PoG---postcentral-gyrus", "Right-POrG--posterior-orbital-gyrus", "Left-POrG--posterior-orbital-gyrus", "Right-PP----planum-polare", "Left-PP----planum-polare", "Right-PrG---precentral-gyrus", "Left-PrG---precentral-gyrus", "Right-PT----planum-temporale", "Left-PT----planum-temporale", "Right-SCA---subcallosal-area", "Left-SCA---subcallosal-area", "Right-SFG---superior-frontal-gyrus", "Left-SFG---superior-frontal-gyrus", "Right-SMC---supplementary-motor-cortex", "Left-SMC---supplementary-motor-cortex", "Right-SMG---supramarginal-gyrus", "Left-SMG---supramarginal-gyrus", "Right-SOG---superior-occipital-gyrus", "Left-SOG---superior-occipital-gyrus", "Right-SPL---superior-parietal-lobule", "Left-SPL---superior-parietal-lobule", "Right-STG---superior-temporal-gyrus", "Left-STG---superior-temporal-gyrus", "Right-TMP---temporal-pole", "Left-TMP---temporal-pole", "Right-TrIFG-triangular-part-of-the-IFG", "Left-TrIFG-triangular-part-of-the-IFG", "Right-TTG---transverse-temporal-gyrus", "Left-TTG---transverse-temporal-gyrus"]

    batch_size = eval_bulk['batch_size']
    need_unc = eval_bulk['estimate_uncertainty']
    mc_samples = eval_bulk['mc_samples']
    dir_struct = eval_bulk['directory_struct']
    if 'exit_on_error' in eval_bulk.keys():
        exit_on_error = eval_bulk['exit_on_error']
    else:
        exit_on_error = False

    if eval_bulk['view_agg'] == 'True':
        coronal_model_path = eval_bulk['coronal_model_path']
        axial_model_path = eval_bulk['axial_model_path']
        eu.evaluate2view(coronal_model_path,
                         axial_model_path,
                         volumes_txt_file,
                         data_dir, device,
                         prediction_path,
                         batch_size,
                         label_names,
                         dir_struct,
                         need_unc,
                         mc_samples,
                         exit_on_error=exit_on_error)
    else:
        coronal_model_path = eval_bulk['coronal_model_path']
        eu.evaluate(coronal_model_path,
                    volumes_txt_file,
                    data_dir,
                    device,
                    prediction_path,
                    batch_size,
                    "COR",
                    label_names,
                    dir_struct,
                    need_unc,
                    mc_samples,
                    exit_on_error=exit_on_error)

def compute_vol(eval_bulk):
    prediction_path = eval_bulk['save_predictions_dir']
    label_names = ["vol_ID", "Background", "Left WM", "Left Cortex", "Left Lateral ventricle", "Left Inf LatVentricle",
                   "Left Cerebellum WM", "Left Cerebellum Cortex", "Left Thalamus", "Left Caudate", "Left Putamen",
                   "Left Pallidum", "3rd Ventricle", "4th Ventricle", "Brain Stem", "Left Hippocampus", "Left Amygdala",
                   "CSF (Cranial)", "Left Accumbens", "Left Ventral DC", "Right WM", "Right Cortex",
                   "Right Lateral Ventricle", "Right Inf LatVentricle", "Right Cerebellum WM",
                   "Right Cerebellum Cortex", "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
                   "Right Hippocampus", "Right Amygdala", "Right Accumbens", "Right Ventral DC"]
    volumes_txt_file = eval_bulk['volumes_txt_file']

    eu.compute_vol_bulk(prediction_path, "Linear", label_names, volumes_txt_file)



def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--setting_path', '-sp', required=False, help='optional path to settings_eval.ini')
    args = parser.parse_args()

    settings = Settings('settings.ini')
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], \
                                                                        settings[
                                                                            'NETWORK'], settings['TRAINING'], \
                                                                        settings['EVAL']
    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'eval_bulk':
        logging.basicConfig(filename='error.log')
        if args.setting_path is not None:
            settings_eval = Settings(args.setting_path)
        else:
            settings_eval = Settings('settings_eval.ini')
        evaluate_bulk(settings_eval['EVAL_BULK'])
    elif args.mode == 'clear':
        shutil.rmtree(os.path.join(common_params['exp_dir'], train_params['exp_name']))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params['log_dir'], train_params['exp_name']))
        print("Cleared current log directory successfully!!")

    elif args.mode == 'clear-all':
        delete_contents(common_params['exp_dir'])
        print("Cleared experiments directory successfully!!")
        delete_contents(common_params['log_dir'])
        print("Cleared logs directory successfully!!")

    elif args.mode == 'compute_vol':
        if args.setting_path is not None:
            settings_eval = Settings(args.setting_path)
        else:
            settings_eval = Settings('settings_eval.ini')
        compute_vol(settings_eval['EVAL_BULK'])
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')
