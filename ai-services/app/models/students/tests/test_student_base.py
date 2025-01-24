import pytest
import torch
from app.models.students.student_base import StudentBase
from torch import nn


class DummyModel(nn.Module):
    """
    A simple dummy model for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def dummy_model():
    """
    Fixture for creating a dummy model instance.
    """
    return DummyModel()


@pytest.fixture
def student_base(dummy_model):
    """
    Fixture for creating a StudentBase instance.
    """
    return StudentBase(dummy_model)


def test_student_base_initialization(dummy_model):
    """
    Test the initialization of the StudentBase class.
    """
    student = StudentBase(dummy_model)
    assert isinstance(student.model, nn.Module)


def test_student_base_initialization_invalid_model():
    """
    Test that initializing with an invalid model raises a TypeError.
    """
    with pytest.raises(TypeError, match="model must be an instance of torch.nn.Module"):
        StudentBase("invalid_model")


def test_forward_pass_with_valid_input(student_base):
    """
    Test the forward method with valid input.
    """
    input_tensor = torch.randn(2, 10)  # Batch of 2, input size 10
    output = student_base(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 5)  # Output size matches the DummyModel


def test_forward_pass_with_invalid_input(student_base):
    """
    Test that the forward method raises a TypeError for invalid input.
    """
    with pytest.raises(TypeError, match="input_ids must be a torch.Tensor"):
        student_base("invalid_input")


def test_forward_pass_logs_error(dummy_model, caplog):
    """
    Test that an error during the forward pass is logged.
    """

    # Create a model that raises an error
    class FailingModel(nn.Module):
        def forward(self, x):
            raise ValueError("Intentional failure")

    failing_student = StudentBase(FailingModel())

    input_tensor = torch.randn(2, 10)
    with pytest.raises(ValueError, match="Intentional failure"):
        failing_student(input_tensor)

    # Check that the error is logged
    assert "Error during forward pass: Intentional failure" in caplog.text
