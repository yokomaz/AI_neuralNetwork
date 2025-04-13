import numpy as np
from tqdm import tqdm

def cross_entropy(y_pred, y_true, Model, l2_lambda=1e-3, epsilon=1e-12):
    y_pred_clip = np.clip(y_pred, epsilon, 1. - epsilon)
    log_probs = np.where(y_true==1, np.log(y_pred_clip), 0)
    sample_losses = -np.sum(y_true * np.log(y_pred_clip), axis=1)
    cross_entropy_loss = np.mean(sample_losses)
    
    # calculate l2 regularization
    # l2_regularization = 0
    # for layer_name, weights in Model.parameters.items():
    #     if 'weights' in layer_name:
    #         l2_regularization += np.sum(weights ** 2)
    #     else:
    #         continue
    total_loss = cross_entropy_loss # + l2_lambda * l2_regularization
    return total_loss

def validation(model, val_dataloader):
    #label_pred_list = []
    #label_y_list = []
    val_loss = 0
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=True, colour='blue')
    for batch_idx, (batch_x, batch_y) in loop:
        y_pred = model.forward(batch_x)

        val_loss += cross_entropy(y_pred, batch_y, model)
        
        label_pred = np.argmax(y_pred, axis=1)
        label_y = np.argmax(batch_y, axis=1)
        # label_pred_list.append(label_pred)
        # label_y_list.append(label_y)
        
        loop.set_description(f'Val idx: [{batch_idx+1}/{val_dataloader.__len__()}]')
        # loop.set_postfix({f"Val loss"})
    # cross entropy loss over val set
    # val_loss /= val_dataloader.__len__()

    # accuracy over val set
    # label_pred_list = np.concatenate(label_pred_list).reshape(-1, 1)
    # label_y_list = np.concatenate(label_y_list).reshape(-1, 1)
    accuracy = np.sum(label_pred == label_y) / len(label_y)#np.sum(label_pred_list == label_y_list) / len(label_pred_list)
    
    return val_loss, accuracy