    total_count = 0
    correct_count = 0
    precision_base = {}
    precision_correct = {}
    recall_base = {}
    recall_correct = {}
    cur_index = -1
    i=0
    if predictor.__class__.__name__ == "DecisionTree":
        labels = predictor.predict(instances)
    else:
        for instance in instances:
            cur_index += 1
            label = predictor.predict(instance)
            if label == instance._label.label_str:
                correct_count = correct_count + 1
            total_count = total_count + 1
            if label not in precision_correct:
                precision_correct[label] = 0
                precision_base[label] = 0
            if instance._label.label_str == label:
                precision_correct[instance._label.label_str] += 1
            precision_base[label] += 1
            if instance._label.label_str not in recall_correct:
                recall_correct[instance._label.label_str] = 0
                recall_base[instance._label.label_str] = 0
            if instance._label.label_str == label:
                recall_correct[instance._label.label_str] += 1
            recall_base[instance._label.label_str] += 1
            #print(str(label))
