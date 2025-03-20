# import .utils

import torch



def loss_function(label_smoothing=0.1):
    if label_smoothing is not None:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion
    # if user_bnn_head:
    #     return F.cross_entropy(y_p, y_t)
    # else:
    #     return F.cross_entropy(y_p, y_t, label_smoothing=0.1)




