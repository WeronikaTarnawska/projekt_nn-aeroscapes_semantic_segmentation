"""Tests for scripts/download_data.py"""

from pathlib import Path
from unittest.mock import patch

from scripts.download_data import download_and_symlink_aeroscapes


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_symlink")
@patch("os.symlink")
@patch("kagglehub.dataset_download")
def test_successful_download_and_symlink(
    mock_download, mock_symlink, mock_is_symlink, mock_exists
):
    """Test successful download and symlink creation."""
    mock_download.return_value = "/cache/path/dataset"
    mock_is_symlink.return_value = False
    mock_exists.return_value = False
    link_path = Path("/tmp/test_data")

    download_and_symlink_aeroscapes(link_path)

    mock_download.assert_called_once_with("kooaslansefat/uav-segmentation-aeroscapes")
    mock_symlink.assert_called_once_with("/cache/path/dataset", link_path)
