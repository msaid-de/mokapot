"""
These tests verify that the aggregatePsmsToPeptides executable works as expected.
"""
import subprocess
from pathlib import Path

import pytest

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture
def psms_files():
    targets_psms = Path("data", "targets.psms")
    decoys_psms = Path("data", "decoys.psms")
    return targets_psms, decoys_psms


def test_basic_cli(tmp_path, psms_files):
    """Test that basic cli works."""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        psms_files[0],
        "--decoys_psms",
        psms_files[1],
        "--dest_dir",
        tmp_path,
    ]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_keep_decoys(tmp_path, psms_files):
    """Test that --keep_decoys works."""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        psms_files[0],
        "--decoys_psms",
        psms_files[1],
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()


def test_non_default_fdr(tmp_path, psms_files):
    """Test non-defaults"""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        psms_files[0],
        "--decoys_psms",
        psms_files[1],
        "--test_fdr",
        "0.1",
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]

    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()
