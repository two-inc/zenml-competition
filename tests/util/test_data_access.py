from unittest.mock import patch, Mock

from util.data_access import load_data, DEFAULT_OBJECT_NAME, BUCKET_URL


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
