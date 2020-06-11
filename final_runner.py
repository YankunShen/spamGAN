import os
import sys
import csv
import importlib
import spamGAN_train
import texar
import random
import wandb
wandb.init()

BASEDIR = '/home/yankun/spamGAN_output/'

def get_config_file(trp, usp):
    if usp == -1:
        return 'stepGAN_base_config_nogan'
    if usp == 0.0:
        return 'stepGAN_base_config_nounsup'
    if usp == 0.5 or usp == 0.6:
        return 'stepGAN_base_config_smallunsup'
    if usp == 0.7 or usp == 0.8:
        return 'stepGAN_base_config_smallunsup'
    if usp == 0.9 or usp == 1.0:
        return 'stepGAN_base_config_smallunsup'

unsup_revs_path = '/home/yankun/spamGAN_output/unlabeled_review.txt'

train_revs = '/home/yankun/spamGAN_output/train_review.txt'
train_labs = '/home/yankun/spamGAN_output/train_label.txt'
test_revs = '/home/yankun/spamGAN_output/test_review.txt'
test_labs = '/home/yankun/spamGAN_output/test_label.txt'

def make_data(trp, usp, run):
    nogan = False
    if usp == -1:
        usp = 0.0
        nogan = True
    with open(train_revs, 'r') as f:
        revs = f.readlines()
    with open(train_labs, 'r') as f:
        labs = f.readlines()
    
    shfl_idx = random.sample(list(range(len(revs))), len(revs))
    revs = [str(revs[i]) for i in shfl_idx]
    labs = [str(labs[i]) for i in shfl_idx]


    tr = revs[:round(trp * len(revs) * 0.9)]
    vr = revs[round(0.9 * trp * len(revs)): round(trp * len(revs))]
    tl = labs[:round(trp * len(revs) * 0.9)]
    vl = labs[round(0.9 * trp * len(revs)): round(trp * len(revs))]
 
    if len(vr) == 0 :
        # just add a fake as a workaround
        vr = revs[0:100]
        vl = labs[0:100]
    with open(unsup_revs_path, 'r') as f:
        unsup_revs_full = f.readlines()
    random.shuffle(unsup_revs_full)
    unsup_revs = unsup_revs_full[:round(usp * len(unsup_revs_full))]

    unsup_labs = ['-1\n'] * len(unsup_revs)


    dir_name = 'tr{}_usp{}_{}'.format(int(trp*100), int(usp * 100), run)
    result_name = 'tr{}_usp{}'.format(int(trp*100), int(usp * 100))
    if nogan:
        dir_name = dir_name + '_nogan/'
        result_name = result_name + '_nogan'
    os.mkdir(os.path.join(BASEDIR, dir_name))
    curdir = os.path.join(BASEDIR, dir_name)
    resultdir = os.path.join(BASEDIR, "result")
    result_file = os.path.join(resultdir, result_name)
    data_paths = {
        'train_data_reviews' : os.path.join(curdir, 'trevs.txt'),
        'train_data_labels'  : os.path.join(curdir, 'tlabs.txt'),
        'val_data_reviews' : os.path.join(curdir, 'vrevs.txt'),
        'val_data_labels' : os.path.join(curdir, 'vlabs.txt'),
        'unsup_train_data_reviews' : os.path.join(curdir, 'unsup_trevs.txt'),
        'unsup_train_data_labels' : os.path.join(curdir, 'unsup_tlabs.txt'),
        'vocab' : os.path.join(curdir, 'vocab.txt'),
        'clas_test_ckpt' : os.path.join(curdir, 'ckpt-bestclas'),
        'clas_pred_output' : os.path.join(curdir, 'testpreds.txt'),
        'dir' : curdir,
        'result_file' : result_file,
        'clas_pretrain_save' : nogan
    }


 
    with open(data_paths['train_data_reviews'], 'w') as f: 
        for x in tr: 
            f.write(x)

    with open(data_paths['train_data_labels'], 'w') as f:
        for x in tl:
            f.write(str(x))
 
    with open(data_paths['unsup_train_data_reviews'], 'w') as f: 
        for x in unsup_revs: 
            f.write(x)
  
    with open(data_paths['unsup_train_data_labels'], 'w') as f:
        for x in unsup_labs:
            f.write(str(x))

    with open(data_paths['val_data_reviews'], 'w') as f:
        for x in vr:
            f.write(x)

    with open(data_paths['val_data_labels'], 'w') as f:
        for x in vl:
            f.write(str(x))



    vocab = texar.data.make_vocab([train_revs, test_revs, data_paths['unsup_train_data_reviews']], 10000)

    with open(data_paths['vocab'], 'w') as f:
        for v in vocab:
            f.write(v + '\n')


    return data_paths

# 0.5, 0.8 x 0.5, 0.8
for train_pcent in [1.0]:
    for unsup_pcent in [-1]:
        for run in range(0, 5):
            base_config_file = 'spamGAN_config_smallunsup'
            data_paths = make_data(train_pcent, unsup_pcent, run)
            importlib.invalidate_caches()
            base_config = importlib.import_module(base_config_file)
            base_config = importlib.reload(base_config)
            # inject file paths
            base_config.train_data['datasets'][0]['files'] = [data_paths['train_data_reviews'],
                                                              data_paths['unsup_train_data_reviews']]
            base_config.train_data['datasets'][1]['files' ] = [data_paths['train_data_labels'],
                                                               data_paths['unsup_train_data_labels']]
                                                               
            base_config.clas_train_data['datasets'][0]['files'] = data_paths['train_data_reviews']
            base_config.clas_train_data['datasets'][1]['files'] = data_paths['train_data_labels']
            base_config.val_data['datasets'][0]['files'] = data_paths['val_data_reviews']
            base_config.val_data['datasets'][1]['files'] = data_paths['val_data_labels']
            base_config.test_data['datasets'][0]['files'] = test_revs
            base_config.test_data['datasets'][1]['files'] = test_labs
            base_config.train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.val_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.test_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_test_ckpt = data_paths['clas_test_ckpt']
            base_config.clas_pred_output = data_paths['clas_pred_output']
            base_config.log_dir = data_paths['dir']
            base_config.checkpoint_dir = data_paths['dir']
            print(base_config.train_data['datasets'][0]['files'])
            print('Train Pcent {} Unsup Pcent {} Run {}'.format(train_pcent, unsup_pcent, run))
            # Run
            dict_res = spamGAN_train.main(base_config)
            
            file_exists = os.path.isfile(data_paths["result_file"])
            f = open(data_paths["result_file"],'a')
            w = csv.DictWriter(f, dict_res.keys())
            if not file_exists:
                print("writing header")
                w.writeheader()
            w.writerow(dict_res)
            f.close()



