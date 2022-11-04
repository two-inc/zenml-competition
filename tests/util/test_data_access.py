"""testing data access functions """
from unittest.mock import Mock
from unittest.mock import patch
import pytest

from src.util.data_access import BUCKET_URL
from src.util.data_access import DEFAULT_OBJECT_NAME
from src.util.data_access import load_data, get_data_for_test


def _mock_request(mock_urlopen, response_data):
    """Mock request"""
    mock_response = Mock()
    mock_response.read.return_value = response_data.encode("utf8")
    mock_handle = mock_urlopen.return_value
    mock_handle.__enter__.return_value = mock_response
    mock_handle.__exit__.return_value = False


@patch("urllib.request.urlopen")
def test_load_default_data(mock_urlopen):
    """test_load_default_data"""

    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    data = load_data()

    assert mock_urlopen.call_count == 1
    assert (
        mock_urlopen.call_args[0][0].full_url
        == BUCKET_URL + DEFAULT_OBJECT_NAME
    )
    assert not data.empty


@patch("urllib.request.urlopen")
def test_load_custom_data(mock_urlopen):
    """test_load_custom_data"""
    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    data = load_data("my_custom_object")

    assert mock_urlopen.call_count == 1
    assert (
        mock_urlopen.call_args[0][0].full_url
        == BUCKET_URL + "my_custom_object"
    )
    assert not data.empty


@patch("pandas.read_csv", return_value=None)
@patch("urllib.request.urlopen")
def test_load_default_data_no_data_found(mock_urlopen, read_csv):
    """test_load_default_data"""

    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    with pytest.raises(ValueError, match="Sample data not found"):
        load_data()


@patch("pandas.read_csv", return_value=[1,2,3])
@patch("urllib.request.urlopen")
def test_load_default_data_incorrect_data_type(mock_urlopen, read_csv):
    """test_load_default_data"""

    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    with pytest.raises(ValueError, match="Could not load sample data as a DataFrame. Loaded instead as type list"):
        load_data()


@patch("urllib.request.urlopen")
def test_get_data_for_test(mock_urlopen):
    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    data = get_data_for_test()
    assert len(data) == 100


@patch("pandas.read_csv", return_value=None)
@patch("urllib.request.urlopen")
def test_get_data_for_test_except_block(mock_urlopen, read_csv):
    """testing that the try except block"""
    _mock_request(mock_urlopen, "A,B,C\n1,2,3")

    with pytest.raises(AttributeError, match="NoneType' object has no attribute 'copy'"):
        data = get_data_for_test()
