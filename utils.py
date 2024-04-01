import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def get_col_names(metadata_path):
    with open(metadata_path) as file:
        arff_data = file.read()

    cols = []
    for line in arff_data.splitlines():
        if line.startswith("@attribute"):
            col = line.split(" ")[1].strip("'")
            cols.append(col)
    cols.append("level") # manually add last col
    return cols

def get_numeric_cols(df):
    num_only = []
    for col in df.columns:
        if df[col].dtype != "object":
            num_only.append(col)
    return df[num_only]

def convert_label_to_binary(df, label):
    df[label] = df[label].apply(lambda x: 0 if x == "normal" else 1)
    return df


def plot_custom_confusion_matrix(
    cm, target_names=None, title="Conf. Mat.", figsize=(10, 10), normalize=True, saveit=False, showit=True
):
    """plot confusion matrix as heatmap.

    Args:
        cm (np.array): confusion matrix produced by confusion_matrix() function of sklearn.
        target_names (list-like): name of the classes (labels) in order. (use cm_order of from txmlcommonlibrary.utilities.supervisedml.evaluate_clf).
        title (str, optional): Defaults to 'Conf. Mat.'.
        figsize (tuple, optional): Defaults to (10, 10).
        normalize (bool, optional): Defaults to True.
        saveit (bool, optional): Defaults to False.
        showit (bool, optional): Defaults to True.
    """
    import itertools

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=30)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #     thresh = cm_norm.max() / 1.5 if normalize else cm.max() / 2
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.2f}%\n{:,}".format(100 * cm_norm[i, j], cm[i, j]),
                horizontalalignment="center",
                # color="white" if cm_norm[i, j] > thresh else "black",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.grid(b=None)

    if saveit:
        plt.savefig(fname=saveit)
    if showit:
        plt.show()

def plot_heatmap_masked(corr, figsize=(10, 6), annot=True, title="Correlation Heatmap"):
    """plot reduced heatmap.

    Args:
        corr (pd.DataFrame): correlation dataframe produced via pd.DataFrame.corr().
        figsize (tuple, optional): Defaults to (10, 6).
        annot (bool, optional): annotations. Defaults to True.
        title (str, optional): Defaults to "Correlation Heatmap".
    """
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        center=0,
        square=True,
        annot=annot,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title(title)

def plot_custom_roc_curve(y_true, preds_probs, figsize=(10, 6), title="ROC"):
    """plot ROC curve.
        Args:
        y_true (np.array): true value classes of val/test set. (i.e. ground truth).
        preds_probs (np.array): predicted probabilities on val/test set. (i.e: model.predict_proba(X_val)[:, 1]). Defaults to None.
        figsize (tuple, optional): Defaults to (10, 6).
        title (str, optional): Defaults to "ROC".
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    auc = roc_auc_score(y_true, preds_probs)
    plt.title(f"{title}\nAUC:{auc}")

    # no-skill learner
    ns_probs = np.zeros(y_true.shape[0])
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)

    # model's
    fpr, tpr, _ = roc_curve(y_true, preds_probs)

    plt.plot(fpr, tpr, marker=".")
    plt.plot(ns_fpr, ns_tpr, linestyle="--")
    plt.tight_layout()
    plt.xlabel("FPR")
    plt.ylabel("TPR")

def plot_prediction_distribution(
    y_true,
    preds_probs,
    title="score distribution",
    figsize=(10, 6),
    configs={
        "bins": np.arange(0, 1.1, 0.01),
        "alpha": 0.4,
    },
):
    """plot the distribution of predictions for a binary classifier.

    Args:
        y_true (np.array): true value classes of val/test set. (i.e. ground truth).
        preds_probs (np.array): predicted probabilities on val/test set. (i.e: model.predict_proba(X_val)[:, 1]). Defaults to None.
        title (str, optional): Defaults to "score distribution".
        figsize (tuple, optional): Defaults to (10, 6).
        configs (dict, optional): configurations to be used in plotting. Defaults to { "bins": np.arange(0, 1.1, 0.01), "alpha": 0.4, }.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data = preds_probs[y_true == 0]
    sns.histplot(x=data, ax=ax, kde=True, weights=np.ones(len(data)) / len(data), label=f"Class 0", **configs)

    data = preds_probs[y_true > 0]
    sns.histplot(
        x=data, ax=ax, kde=True, weights=np.ones(len(data)) / len(data), label=f"Class 1", color="red", **configs
    )

    ax.legend()
    ax.set_title(title)


def plot_histogram(data, x, is_excluded, ax, binwidth=10, stat="percent"):
    sns.histplot(
        data=data,
        x=x,
        stat=stat,
        ax=ax,
        binwidth=binwidth,
    )
    ax.set_title("(excluding perfect scores)" if is_excluded else "")
    ax.set_xlabel(f"{x} Trip Score")
    ax.set_ylabel(f"Trips ({stat})")
    ax.tick_params(which="both", bottom=True)
    logging.info("- - - - ")