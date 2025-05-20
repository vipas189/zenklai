def early_stop(model, history, hyperparameters):
    if history.get("epoch_val_acc") > history.get("best_val_accuracy"):
        history["best_val_accuracy"] = history.get("epoch_val_acc")
        history["best_model_wts"] = model.state_dict()
        history["not_improved_val_acc"] = 0

    history["not_improved_val_acc"] += 1

    if history["not_improved_val_acc"] == hyperparameters.get("early_stop"):
        history["not_improved_val_acc"] = 0
        print("Model stopped, no more improvement in validation accuracy")
        return True
