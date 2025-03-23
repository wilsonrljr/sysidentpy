import pytest
import pickle as pk
from sysidentpy.utils.save_load import save_model, load_model


# Sample class to use as a mock model
class MockModel:
    def __init__(self, value):
        self.value = value


def test_save_model(tmp_path):
    model = MockModel(42)  # Create a simple mock model
    file_name = "test_model.syspy"
    file_path = tmp_path / file_name

    # Call the function
    save_model(model=model, file_name=str(file_path))

    # Check if file was created
    assert file_path.exists()

    # Check if model was saved correctly
    with open(file_path, "rb") as f:
        loaded_model = pk.load(f)

    assert isinstance(loaded_model, MockModel)
    assert loaded_model.value == model.value  # Ensure model data is intact


def test_save_model_without_model():
    with pytest.raises(TypeError, match="model cannot be None."):
        save_model(model=None, file_name="should_fail.syspy")


def test_save_model_with_path(tmp_path):
    model = MockModel(99)
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()  # Create a subdirectory

    file_name = "nested_model.syspy"
    file_path = sub_dir / file_name

    save_model(model=model, file_name=file_name, path=str(sub_dir))

    assert file_path.exists()

    # Verify saved model content
    with open(file_path, "rb") as f:
        loaded_model = pk.load(f)

    assert loaded_model.value == model.value


def test_load_model(tmp_path):
    model = MockModel(42)  # Create a simple mock model
    file_name = "test_model.syspy"
    file_path = tmp_path / file_name

    # Save the model first
    save_model(model=model, file_name=str(file_path))

    # Load the model
    loaded_model = load_model(file_name=str(file_path))

    # Check if the loaded model is correct
    assert isinstance(loaded_model, MockModel)
    assert loaded_model.value == model.value  # Ensure model data is intact


def test_load_model_with_path(tmp_path):
    model = MockModel(99)
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()

    file_name = "nested_model.syspy"

    save_model(model=model, file_name=file_name, path=str(sub_dir))

    # Load the model specifying the path
    loaded_model = load_model(file_name=file_name, path=str(sub_dir))

    assert isinstance(loaded_model, MockModel)
    assert loaded_model.value == model.value


def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model(file_name="non_existent_model.syspy")
