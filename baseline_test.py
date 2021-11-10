import json
import argparse
import torch
import numpy as np
from lib import metrics
import model.dcrnn_model as module_arch
from parse_config import ConfigParser
from lib import baseline_utils as utils
from tqdm import tqdm
import math
import time


def calc_mape_np(pred,gnd):
    msk = gnd > 5
    pred = pred[msk]
    gnd = gnd[msk]
    abs = np.abs(pred - gnd)
    abs /= gnd
    return np.mean(abs)

def calc_rmse_np(pred,gnd):
    msk = gnd > 5
    gnd = gnd[msk]
    pred = pred[msk]
    return np.sqrt(np.mean((pred - gnd)**2))


def main(config):
    logger = config.get_logger('test')
    dataset = 'round2'
    graph_pkl_filename = 'data/{}_withInOut_splitlength.pkl'.format(dataset)
    adj_mat = utils.load_graph_data(graph_pkl_filename)
    data, means, stds = utils.load_dataset(data_path='data/{}_withInOut_splitlength.pkl'.format(dataset),
                              batch_size=config["arch"]["args"]["batch_size"],
                              test_batch_size=config["arch"]["args"]["batch_size"])
    test_data_loader = data['test_loader']
    scaler = data['scaler']
    num_test_iteration= math.ceil(data['x_test'].shape[0] / config["arch"]["args"]["batch_size"])

    num_test_sample = data['x_test'].shape[0]
    config["arch"]["args"]["num_nodes"] = data['x_test'].shape[2]

    # build model architecture
    adj_arg = {"adj_mat": adj_mat}
    model = config.initialize('arch', module_arch, **adj_arg)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_preds = torch.FloatTensor([])
    y_truths = data['y_test']  # (batch 10, step 1, nodes 18072,  feat 1)
    y_truths = scaler.inverse_transform(y_truths)
    predictions = []
    groundtruth = list()

    start_time = time.time()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_data_loader.get_iterator()), total=num_test_iteration):
            # x = torch.FloatTensor(x).cuda()
            x = torch.FloatTensor(x)
            # y = torch.FloatTensor(y).cuda()
            y = torch.FloatTensor(y)
            

            outputs = model(x, y, 0)  # (seq_length, batch_size, num_nodes*output_dim)  (10, 50, 207*1)
            outputs = torch.transpose(outputs.view(1, model.batch_size, model.num_nodes,
                                                 model.output_dim),0,1  )# back to (10, 1, 207, 1)

            xx = torch.transpose(x[0], 0 ,1).squeeze(2)
            yy = torch.transpose(y[0], 0, 1).squeeze(2).squeeze(1)
            outputss = torch.transpose(outputs[0],0 ,1).squeeze(2).squeeze(1)
            
            xx = scaler.inverse_transform(xx)
            yy =  scaler.inverse_transform(yy)
            outputss =  scaler.inverse_transform(outputss)
            msk = yy > 50
            xx = xx[msk]
            yy = yy[msk]
            outputss = outputss[msk]
            for id, xxx in enumerate(xx):
                print(xx[id],yy[id],outputss[id])
            # print("x:{},{}\ny:{},{}\noutputs:{},{}".format(np.shape(x),x,np.shape(y),y,np.shape(outputs),outputs))
            y_preds = torch.cat([y_preds, outputs], dim=1)
    inference_time = time.time() - start_time
    logger.info("Inference time: {:.4f} s".format(inference_time))
    # y_preds = torch.transpose(y_preds, 0, 1)
    # outputs = torch.transpose(outputs, 0, 1)
    y_preds = y_preds.detach().numpy()  # cast to numpy array
    print(np.shape(y_truths), np.shape(y_preds))
    print("--------test results--------")
    for horizon_i in range(y_truths.shape[1]):
        y_truth = np.squeeze(y_truths[:, horizon_i, :, 0])
        y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
        # y_pred = y_preds[:,horizon_i,:]
        print(y_truth,y_pred)
        print(np.shape(y_truth),np.shape(y_pred))
        # mask = (y_truth * stds + means > 5)
        predictions.append(y_pred)
        groundtruth.append(y_truth)
        mask = y_truth > 5
        print(mask*y_pred[:y_truth.shape[0]])
        print(mask*y_truth)

        big_msk = y_truth > 20
        print(y_pred[:y_truth.shape[0]][big_msk])
        print(y_truth[big_msk])

        mae = metrics.masked_mae_np(mask*y_pred[:y_truth.shape[0]], mask*y_truth, null_val=0)
        mape = metrics.masked_mape_np(mask*y_pred[:y_truth.shape[0]], mask*y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(mask*y_pred[:y_truth.shape[0]], mask*y_truth, null_val=0)
        mape_mine = calc_mape_np(mask*y_pred[:y_truth.shape[0]],mask*y_truth)
        rmse_mine = calc_rmse_np(mask*y_pred[:y_truth.shape[0]],mask*y_truth)
        print(
            "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}, MAPE_MINE: {:.4f}, RMSE_MINE: {:.4f}".format(
                horizon_i + 1, mae, mape, rmse,mape_mine,rmse_mine
            )
        )
        log = {"Horizon": horizon_i+1, "MAE": mae, "MAPE": mape, "RMSE": rmse, "MAPE_MINE": mape_mine, "RMSE_MINE": rmse_mine}
        logger.info(log)
    outputs = {
        'predictions': predictions,
        'groundtruth': groundtruth
    }
    json.dump(outputs,open('./test_out.json','w'),indent=2)
    # serialize test data
    # np.savez_compressed('saved/results/dcrnn_predictions.npz', **outputs)
    # print('Predictions saved as {}.'.format('saved/results/dcrnn_predictions.npz'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch DCRNN')

    args.add_argument('-r', '--resume', default='/root/pytorch-DCRNN/saved/round2/models/round2_DCRNN/1109_223402/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
