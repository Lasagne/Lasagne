from __future__ import absolute_import
from mock import Mock
import pytest


@pytest.fixture
def dummy_input_layer():
    input_layer = Mock()
    input_layer.get_output_shape.return_value = (2, 3, 4)
    return input_layer
