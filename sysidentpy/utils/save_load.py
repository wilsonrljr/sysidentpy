# Author:
#           Samir Angelo Milani Martins https://github.com/samirmartins
# License: BSD 3 clause

import pickle as pk
import os


def save_model(
    *,
    model=None,
    file_name="model",
    path=None,
):
    """Save the model "model" in folder "folder" using an extension .syspy.

    Parameters
    ----------
    model: the model variable to be saved
    file_name: file name, along with .syspy extension
    path: location where the model will be saved (optional)

    Returns
    -------
    file file_name.syspy located at "path", containing the estimated model.

    """
    if model is None:
        raise TypeError("model cannot be None.")

    # Checking if path is provided
    if path is not None:

        # Composing file_name with path
        file_name = os.path.join(path, file_name)

    # Saving model
    with open(file_name, "wb") as fp:
        pk.dump(model, fp)


def load_model(
    *,
    file_name="model",
    path=None,
):
    """Load the model from file "file_name.syspy" located at path "path".

    Parameters
    ----------
    file_name: file name (str), along with .syspy extension of the file containing
        model to be loaded
    path: location where "file_name.syspy" is (optional).

    Returns
    -------
    model_loaded: model loaded, as a variable, containing model and its attributes

    """
    # Checking if path is provided
    if path is not None:

        # Composing file_name with path
        file_name = os.path.join(path, file_name)

    # Loading the model
    with open(file_name, "rb") as fp:
        model_loaded = pk.load(fp)

    return model_loaded
