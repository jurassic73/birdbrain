"""The service's pure helpers: the ring buffer, redaction, drawing and the CPU figure."""
import numpy as np

from birdbrain.hailo import geometry, service


def test_ring_buffer_keeps_the_last_seconds_and_finds_the_nearest_frame():
    ring = service.RingBuffer(seconds=1.0, fps=10)
    for i in range(30):
        ring.push(i * 0.1, np.full((2, 2, 3), i, np.uint8))
    assert len(ring.frames) == 12  # 1 s at 10 fps plus two spare
    t, frame = ring.nearest(2.0)
    assert abs(t - 2.0) < 1e-6 and int(frame[0, 0, 0]) == 20
    t, _ = ring.nearest(-5.0)  # before the buffer: its oldest frame
    assert t == ring.frames[0][0]
    assert service.RingBuffer(3, 15).nearest(0.0) is None


def test_redact_hides_credentials_in_an_rtsp_url():
    assert service.redact("watching rtsp://user:p%40ss@cam.local:554/ch1") == "watching rtsp://***@cam.local:554/ch1"
    assert service.redact("no url here") == "no url here"


def test_draw_returns_a_copy_with_the_box_on_it():
    frame = np.zeros((36, 64, 3), np.uint8)
    out = service.draw(frame, geometry.Detection("bird", 0.87, (0.25, 0.25, 0.75, 0.75)), "best frame")
    assert out.shape == frame.shape and frame.max() == 0  # the original is untouched
    assert out.max() > 0
    assert service.draw(frame, None).max() == 0  # nothing to draw, still a copy


def test_jpeg_encodes_a_frame():
    data = service.jpeg(np.zeros((36, 64, 3), np.uint8))
    assert data[:2] == b"\xff\xd8"


def test_cpu_percent_is_of_one_core():
    assert service.cpu_percent((10.0, 100.0), (12.5, 110.0)) == 25.0
    assert service.cpu_percent((10.0, 100.0), (12.5, 100.0)) == 0.0
