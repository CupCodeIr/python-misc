import torch


def binary_classification_evaluation(logits : torch.Tensor, activation_function, threshold, targets):
    
    # Checking for arguments type criteria
    logits_valid = torch.is_floating_point(logits)
    threshold_valid = torch.is_tensor(threshold)
    targets_valid = torch.is_tensor(targets) and targets.dtype == torch.bool
    
    # Raise TypeError if at least one of the validity checks fail
    if not logits_valid or not threshold_valid or not targets_valid:
        raise TypeError("There is a problem in arguments type!")
    
    # Check if logits and targets are in shape (n_samples,)
    try:
        shapes_valid = logits.shape[1]
        shapes_valid = targets.shape[1]
        shapes_valid = False
    except:
        shapes_valid = True
    
    # Check if targets has at least one True and one False value
    true_false_comb_check = not torch.all(targets == True) and not torch.all(targets == False)

    # Raise ValueError if logits and targets shapes are not equal or targets does not consist of both True False values
    if not shapes_valid or not logits.shape == targets.shape or not true_false_comb_check:
        raise ValueError("There is a problem in logits or targets values!")

    sample_len = len(targets)

    # Applying the activation function to the model output
    nn_outputs = activation_function(logits)

    # Getting actual predictions by applying threshold
    nn_predictions = (nn_outputs >= threshold).long()

    # Finding TP, TN, FP, FN counts
    tp_count = fn_count = fp_count = tn_count = 0
    for index, pred in enumerate(nn_predictions):
        if(pred == 1 and targets[index] == 1):
            tp_count += 1

        if(pred == 0 and targets[index] == 1):
            fn_count += 1

        if(pred == 1 and targets[index] == 0):
            fp_count += 1

        if(pred == 0 and targets[index] == 0):
            tn_count += 1

    confusion_matrix = [[tp_count, fn_count], [fp_count, tn_count]]
    accuracy = float((tp_count + tn_count) / sample_len)
    precision = weird_division(tp_count, (tp_count + fp_count))
    recall = weird_division(tp_count, (tp_count + fn_count))
    f1_score = float(weird_division(2* precision * recall, precision + recall))
    selectivity = float(weird_division(tn_count, tn_count + fp_count))
    b_accuracy = (recall + selectivity) / 2

    return confusion_matrix, f1_score, accuracy, b_accuracy

def weird_division(a, b):
    # Return zero in case of division by zero
    return a / b if b else 0