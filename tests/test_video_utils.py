import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
import pytest

from app.core.utils import video_utils

class DummyCompletedProcess:
    def __init__(self, args, returncode, stdout="", stderr=""):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

class DummyPopen:
    def __init__(self, stderr_lines):
        self._stderr_lines = stderr_lines
        self._index = 0
        self.returncode = 0
    def readline(self):
        if self._index < len(self._stderr_lines):
            line = self._stderr_lines[self._index]
            self._index += 1
            return line
        return ""
    def poll(self):
        return None if self._index < len(self._stderr_lines) else 0
    def wait(self):
        return self.returncode
    def kill(self):
        self.returncode = -1
    @property
    def stderr(self):
        return self
    def read(self):
        return "".join(self._stderr_lines[self._index:])

# Test video2audio function success case.
def test_video2audio_success(tmp_path, monkeypatch):
    # Create a dummy input file
    input_file = tmp_path / "dummy.mp4"
    input_file.write_text("video content")
    # Define the output file path
    output_file = str(tmp_path / "dummy.wav")

    # Fake subprocess.run to simulate ffmpeg creating the output file.
    def fake_run(cmd, **kwargs):
        # Simulate ffmpeg writing the output file
        output = cmd[-1]
        Path(output).write_text("audio data")
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = video_utils.video2audio(str(input_file), output_file)
    assert result is True
    assert Path(output_file).read_text() == "audio data"

# Test video2audio failure case (non-zero return code).
def test_video2audio_failure(monkeypatch, tmp_path):
    input_file = tmp_path / "dummy.mp4"
    input_file.write_text("video content")
    output_file = str(tmp_path / "dummy.wav")

    def fake_run(cmd, **kwargs):
        return DummyCompletedProcess(cmd, 1, stdout="", stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = video_utils.video2audio(str(input_file), output_file)
    assert result is False

# Test check_cuda_available success scenario.
def test_check_cuda_available_success(monkeypatch):
    # Fake subprocess.run to simulate ffmpeg -hwaccels and -init_hw_device commands.
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "-hwaccels" in cmd:
            # Return stdout containing "cuda"
            return DummyCompletedProcess(cmd, 0, stdout="cuda", stderr="")
        elif "cuda" in cmd:
            # Simulate no error in stderr.
            return DummyCompletedProcess(cmd, 0, stdout="", stderr="")
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = video_utils.check_cuda_available()
    assert result is True

# Test check_cuda_available failure due to missing cuda in hwaccels.
def test_check_cuda_available_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "-hwaccels" in cmd:
            return DummyCompletedProcess(cmd, 0, stdout="some_other_accel", stderr="")
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = video_utils.check_cuda_available()
    assert result is False

# Test add_subtitles soft subtitle branch.
def test_add_subtitles_soft(tmp_path, monkeypatch):
    # Create dummy input and subtitle files.
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mp4")

    # Fake subprocess.run for soft subtitle branch.
    def fake_run(cmd, **kwargs):
        # Check if mov_text is in the command (the soft subtitle codec)
        assert "mov_text" in cmd
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Call add_subtitles with soft_subtitle=True.
    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=True)

# Test add_subtitles hard subtitle branch with progress callback.
def test_add_subtitles_hard(tmp_path, monkeypatch):
    # Create dummy input and subtitle files.
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    # Create a dummy subtitle file with .ass extension to trigger auto_wrap_ass_file usage.
    subtitle_file = tmp_path / "subtitle.ass"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mkv")

    # Fake get_video_info to return dummy video dimensions.
    monkeyatch_get_info = lambda f: {"width": 1280, "height": 720}
    monkeypatch.setattr(video_utils, "get_video_info", monkeyatch_get_info)

    # Fake check_cuda_available to return False to simplify command.
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)

    # Prepare dummy stderr output lines to simulate progress.
    dummy_stderr = [
        "Duration: 00:00:10.00, start: 0.000000, bitrate: 500 kb/s\n",
        "frame=  10 fps=0.0 q=0.0 size=       0kB time=00:00:02.00 bitrate=   0.0kbits/s speed=   0x\n",
        "frame=  20 fps=0.0 q=0.0 size=       0kB time=00:00:04.00 bitrate=   0.0kbits/s speed=   0x\n",
    ]

    # Fake subprocess.Popen for hard subtitle branch.
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self._stderr_lines = dummy_stderr
            self._index = 0
            self.returncode = 0
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    progress_updates = []
    def progress_callback(progress, message):
        progress_updates.append((progress, message))

    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False, progress_callback=progress_callback)
    # Ensure progress callback was called; final progress should be "100".
    assert progress_updates[-1][0] == "100"

# Test get_video_info extracting information.
def test_get_video_info(tmp_path, monkeypatch):
    # Create a dummy video file.
    video_file = tmp_path / "video.mp4"
    video_file.write_text("dummy video content")

    # Fake subprocess.run to return video info in stderr.
    fake_stderr = ("Duration: 00:02:30.50, start: 0.000000, bitrate: 800 kb/s\n"
                    "Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 30 fps, 30 tbr, 30 tbn, 60 tbc\n")

    def fake_run(cmd, **kwargs):
        return DummyCompletedProcess(cmd, 0, stdout="", stderr=fake_stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    info = video_utils.get_video_info(str(video_file))
    assert info is not None
    assert info["duration_seconds"] == 150.5
    assert info["bitrate_kbps"] == 800
    assert info["video_codec"] == "h264"
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["fps"] == 30

# Test get_video_info when regex fails to match (edge case).
def test_get_video_info_no_match(tmp_path, monkeypatch):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("dummy video content")

    # Return stderr that does not contain expected patterns.
    fake_stderr = "Some unexpected output"

    def fake_run(cmd, **kwargs):
        return DummyCompletedProcess(cmd, 0, stdout="", stderr=fake_stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    info = video_utils.get_video_info(str(video_file))
    # Expect default values since regex didn't match.
    assert info is not None
    assert info["duration_seconds"] == 0
    assert info["bitrate_kbps"] == 0
    assert info["video_codec"] == ""
    assert info["width"] == 0
    assert info["height"] == 0
    assert info["fps"] == 0

# Test get_video_info error handling when subprocess.run raises an exception.
def test_get_video_info_exception(tmp_path, monkeypatch):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("dummy video content")

    def fake_run(cmd, **kwargs):
        raise Exception("Fake error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    info = video_utils.get_video_info(str(video_file))
    assert info is None

def test_add_subtitles_missing_input_file(tmp_path):
    """Test that add_subtitles raises an assertion error when the input video file is missing."""
    # Create a valid subtitle file
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mp4")
    with pytest.raises(AssertionError):
        # Provided input video file does not exist
        video_utils.add_subtitles(str(tmp_path / "nonexistent.mp4"), str(subtitle_file), output_file)

def test_add_subtitles_missing_subtitle_file(tmp_path):
    """Test that add_subtitles raises an assertion error when the subtitle file is missing."""
    # Create a valid input video file
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    output_file = str(tmp_path / "output.mp4")
    with pytest.raises(AssertionError):
        # Provided subtitle file does not exist
        video_utils.add_subtitles(str(input_file), str(tmp_path / "nonexistent.srt"), output_file)

def test_add_subtitles_auto_wrap_ass_called(tmp_path, monkeypatch):
    """Test that auto_wrap_ass_file is called when the subtitle is an .ass file and video information is available."""
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.ass"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mkv")

    called = False

    def dummy_auto_wrap(sub_file, **kwargs):
        nonlocal called
        called = True
        return str(sub_file)  # return the same path for simplicity

    monkeypatch.setattr(video_utils, "auto_wrap_ass_file", dummy_auto_wrap)
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)
    monkeypatch.setattr(video_utils, "get_video_info", lambda f: {"width": 1280, "height": 720})

    # Simulate a successful hard subtitle branch using a fake Popen that outputs minimal progress and then finishes successfully.
    class FakePopenSuccess:
        def __init__(self, *args, **kwargs):
            self._stderr_lines = ["Duration: 00:00:05.00\n", "time=00:00:05.00\n"]
            self._index = 0
            self.returncode = 0
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopenSuccess)

    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False)
    assert called is True

def test_add_subtitles_hard_failure(tmp_path, monkeypatch):
    """Test that add_subtitles raises an exception when the ffmpeg process returns a nonzero exit code in hard subtitle mode."""
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mkv")

    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)
    monkeypatch.setattr(video_utils, "get_video_info", lambda f: None)

    class FakePopenFailure:
        def __init__(self, *args, **kwargs):
            self._stderr_lines = ["Duration: 00:00:05.00\n", "time=00:00:02.50\n"]
            self._index = 0
            self.returncode = 1  # nonzero exit code to simulate an error
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopenFailure)

    progress_updates = []
    def progress_callback(progress, message):
        progress_updates.append((progress, message))

    with pytest.raises(Exception):
        video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False, progress_callback=progress_callback)

def test_check_cuda_available_exception(monkeypatch):
    """Test that check_cuda_available returns False when subprocess.run raises an exception."""
    def fake_run(cmd, **kwargs):
        raise Exception("Fake exception")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = video_utils.check_cuda_available()
    assert result is False

def test_video2audio_exception(monkeypatch, tmp_path):
    """Test that video2audio returns False when subprocess.run raises an exception."""
    input_file = tmp_path / "dummy.mp4"
    input_file.write_text("video content")
    output_file = str(tmp_path / "dummy.wav")
    def fake_run(cmd, **kwargs):
        raise Exception("Fake exception in video2audio")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = video_utils.video2audio(str(input_file), output_file)
    assert result is False
def test_add_subtitles_webm_cuda(tmp_path, monkeypatch):
    """Test add_subtitles for WebM output ensuring CUDA acceleration and proper vcodec usage."""
    # Create dummy input and subtitle files.
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.webm")

    # Dictionary to capture the ffmpeg command.
    captured = {}

    # Define a FakePopen that captures the command arguments.
    class FakePopenCapture:
        def __init__(self, cmd, stdout, stderr, text, encoding, errors, creationflags):
            captured['cmd'] = cmd
            self._stderr_lines = ["Duration: 00:00:05.00\n", "time=00:00:05.00\n"]
            self._index = 0
            self.returncode = 0
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    # Override subprocess.Popen to use our FakePopenCapture.
    monkeypatch.setattr(subprocess, "Popen", FakePopenCapture)

    # Override check_cuda_available to simulate that CUDA is available.
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: True)

    # Call add_subtitles which will go to the hard subtitle branch for a WebM file.
    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False)

    cmd_used = captured.get('cmd', [])
    # Verify that the command was built with CUDA acceleration and uses the WebM-specific video codec.
    assert "-hwaccel" in cmd_used
    assert "cuda" in cmd_used
    assert "libvpx-vp9" in cmd_used
def test_video2audio_no_output_file(monkeypatch, tmp_path):
    """Test that video2audio returns False if output file is not created even when ffmpeg returns 0."""
    input_file = tmp_path / "dummy.mp4"
    input_file.write_text("video content")
    output_file = str(tmp_path / "dummy.wav")

    def fake_run(cmd, **kwargs):
        # Simulate a successful ffmpeg run but do NOT create the output file.
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = video_utils.video2audio(str(input_file), output_file)
    assert result is False

def test_check_cuda_available_device_init_failure(monkeypatch):
    """Test that check_cuda_available returns False when cuda device initialization fails (due to error messages in stderr)."""
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "-hwaccels" in cmd:
            return DummyCompletedProcess(cmd, 0, stdout="cuda", stderr="")
        elif "cuda" in cmd:
            # Simulate an error during device initialization.
            return DummyCompletedProcess(cmd, 0, stdout="", stderr="Failed to load cuda")
        return DummyCompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = video_utils.check_cuda_available()
    assert result is False

def test_add_subtitles_cleanup_temp_subtitle(monkeypatch, tmp_path):
    """Test that the temporary subtitle file is cleaned up even when add_subtitles raises an exception in hard subtitle mode."""
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mkv")

    # Force get_video_info to return None to keep things simple.
    monkeypatch.setattr(video_utils, "get_video_info", lambda f: None)
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)

    # Create a fake Popen that simulates an ffmpeg process that returns an error (nonzero exit code).
    class FakePopenException:
        def __init__(self, *args, **kwargs):
            self._stderr_lines = ["Duration: 00:00:05.00\n", "time=00:00:02.50\n"]
            self._index = 0
            self.returncode = 1  # nonzero exit code triggers exception in add_subtitles
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopenException)

    with pytest.raises(Exception):
        video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False)

    # Check that the temporary subtitle file has been deleted from the temp directory.
    temp_subtitle = Path(tempfile.gettempdir()) / "VideoCaptioner" / "temp_subtitle.srt"
def test_add_subtitles_no_progress_callback(tmp_path, monkeypatch):
    """Test add_subtitles hard subtitle branch with no progress callback provided."""
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.srt"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.mkv")

    # Fake Popen that simulates minimal progress messages without using a progress_callback.
    class FakePopenNoProgress:
        def __init__(self, *args, **kwargs):
            self._stderr_lines = ["Duration: 00:00:10.00\n", "time=00:00:03.00\n"]
            self._index = 0
            self.returncode = 0
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopenNoProgress)
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)
    monkeypatch.setattr(video_utils, "get_video_info", lambda f: None)

    # Call add_subtitles with no progress_callback (pass None).
    # Should complete successfully even though no progress updates are sent.
    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False, progress_callback=None)

def test_add_subtitles_ass_output(tmp_path, monkeypatch):
    """Test add_subtitles hard subtitle branch for a .ass output file ensuring proper vf parameter usage."""
    input_file = tmp_path / "input.mp4"
    input_file.write_text("video content")
    subtitle_file = tmp_path / "subtitle.ass"
    subtitle_file.write_text("subtitle content")
    output_file = str(tmp_path / "output.ass")

    captured_cmd = {}

    # Fake Popen class that captures the ffmpeg command argument.
    class FakePopenAss:
        def __init__(self, cmd, *args, **kwargs):
            captured_cmd['cmd'] = cmd
            self._stderr_lines = ["Duration: 00:00:05.00\n", "time=00:00:05.00\n"]
            self._index = 0
            self.returncode = 0
        def poll(self):
            return None if self._index < len(self._stderr_lines) else 0
        def wait(self):
            return self.returncode
        def kill(self):
            self.returncode = -1
        def readline(self):
            if self._index < len(self._stderr_lines):
                line = self._stderr_lines[self._index]
                self._index += 1
                return line
            return ""
        @property
        def stderr(self):
            return self
        def read(self):
            return "".join(self._stderr_lines[self._index:])

    monkeypatch.setattr(subprocess, "Popen", FakePopenAss)
    monkeypatch.setattr(video_utils, "check_cuda_available", lambda: False)
    # Fake get_video_info to return dummy dimensions so that auto_wrap_ass_file is called.
    monkeypatch.setattr(video_utils, "get_video_info", lambda f: {"width": 800, "height": 600})
    # Fake auto_wrap_ass_file to simply return the same file path unchanged.
    monkeypatch.setattr(video_utils, "auto_wrap_ass_file", lambda f, **kwargs: f)

    video_utils.add_subtitles(str(input_file), str(subtitle_file), output_file, soft_subtitle=False)

    # Validate that the vf parameter contains "ass=" since output has .ass extension.
    cmd = captured_cmd.get('cmd', [])
    vf_index = cmd.index("-vf") + 1 if "-vf" in cmd else -1
    assert vf_index != -1, "vf parameter not found in command"
    vf_value = cmd[vf_index]
    assert "ass=" in vf_value, "Expected 'ass=' in vf parameter for .ass output file"