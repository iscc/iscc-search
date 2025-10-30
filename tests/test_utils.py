"""Tests for iscc_search.utils module."""

import time
from io import StringIO
from loguru import logger
from iscc_search.utils import timer


def test_timer_basic_usage():
    # type: () -> None
    """Test timer context manager with default settings."""
    # Capture log output
    log_output = StringIO()
    logger.add(log_output, format="{message}")

    with timer("Test operation"):
        time.sleep(0.01)  # Sleep for at least 10ms

    # Check that completion message was logged
    log_content = log_output.getvalue()
    assert "Test operation - completed" in log_content
    assert "seconds)" in log_content

    # Verify elapsed time is reasonable (at least 0.01 seconds)
    # Extract the time value from log (format: "... (0.XXXX seconds)")
    time_part = log_content.split("(")[1].split(" ")[0]
    elapsed = float(time_part)
    assert elapsed >= 0.01

    # Clean up logger
    logger.remove()


def test_timer_with_log_start():
    # type: () -> None
    """Test timer context manager with log_start enabled."""
    # Capture log output
    log_output = StringIO()
    logger.add(log_output, format="{message}")

    with timer("Operation with start log", log_start=True):
        time.sleep(0.01)

    log_content = log_output.getvalue()

    # Check that both start and completion messages were logged
    assert "Operation with start log - started" in log_content
    assert "Operation with start log - completed" in log_content
    assert "seconds)" in log_content

    # Clean up logger
    logger.remove()


def test_timer_without_log_start():
    # type: () -> None
    """Test timer context manager with log_start disabled (default)."""
    # Capture log output
    log_output = StringIO()
    logger.add(log_output, format="{message}")

    with timer("Operation without start log", log_start=False):
        time.sleep(0.01)

    log_content = log_output.getvalue()

    # Check that start message was NOT logged
    assert "Operation without start log - started" not in log_content

    # Check that completion message WAS logged
    assert "Operation without start log - completed" in log_content
    assert "seconds)" in log_content

    # Clean up logger
    logger.remove()


def test_timer_measures_actual_time():
    # type: () -> None
    """Test that timer accurately measures elapsed time."""
    # Capture log output
    log_output = StringIO()
    logger.add(log_output, format="{message}")

    sleep_duration = 0.05  # 50ms
    with timer("Timing test"):
        time.sleep(sleep_duration)

    log_content = log_output.getvalue()

    # Extract elapsed time from log
    time_part = log_content.split("(")[1].split(" ")[0]
    elapsed = float(time_part)

    # Verify elapsed time is close to sleep_duration (within reasonable margin)
    assert elapsed >= sleep_duration
    assert elapsed < sleep_duration + 0.1  # Allow 100ms overhead

    # Clean up logger
    logger.remove()
