import sys
import logging
import argparse

from model.baseline_jepa import TrajJEPATrainer
from config import Config
from utils import tool_funcs


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    
    parser = argparse.ArgumentParser(description = "HiT-JEPA/train.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--exec_layer', type=int)
    parser.add_argument('--checkpoint_file', type=str, help = '')
    parser.add_argument('--trajcl_training_lr', type=float, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python train.py --dataset porto &> result &
if __name__ == '__main__':
    Config.update(parse_args())
    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/log/'+tool_funcs.log_file_name(), mode = 'w'),
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    T_JEPA = TrajJEPATrainer()
    T_JEPA.train()
    # T_JEPA.eval()
    # T_JEPA.test(Config.exec_layer)

    