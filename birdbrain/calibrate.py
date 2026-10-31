"""Draw the motion ring, crop box and ground zone on a frame to check them before saving to .env.

python -m birdbrain.calibrate out.jpg [--image frame.jpg] [--ring cx,cy,orx,ory,irx,iry] [--crop x,y,w,h]
                                      [--ground "x,y;x,y;..."] [--ground-crop x,y,w,h]
Without --image a fresh frame is grabbed from the main stream; without the other options the .env values are used.
Ring green, crop orange, ground zone blue, ground crop yellow.
"""

import argparse
import subprocess

import cv2
import numpy as np

from . import config
from .watch import ground_mask, motion_mask


def grab() -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", config.rtsp_url(config.RTSP_MAIN),
           "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
    jpeg = subprocess.run(cmd, capture_output=True, timeout=20).stdout
    if not jpeg:
        raise SystemExit("could not grab a frame from the camera")
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--image")
    parser.add_argument("--ring")
    parser.add_argument("--crop")
    parser.add_argument("--spots", help='"cx,cy,rx,ry;cx,cy,rx,ry"')
    parser.add_argument("--ground", help='"x,y;x,y;x,y;..." corners of the ground zone')
    parser.add_argument("--ground-crop")
    args = parser.parse_args()

    try:
        ring = config.parse_ring(args.ring or "", "--ring") or config.MOTION_RING
        crop = config.parse_box(args.crop or "", "--crop") or config.CROP_ROI
        spots = config.parse_spots(args.spots, "--spots") if args.spots else config.MOTION_SPOTS
        ground = config.parse_polygon(args.ground, "--ground") if args.ground else config.GROUND_ZONE
        ground_crop = (config.parse_box(args.ground_crop or "", "--ground-crop")
                       or (config.polygon_box(ground) if args.ground else config.GROUND_CROP))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    frame = cv2.imread(args.image) if args.image else grab()
    H, W = frame.shape[:2]
    out = frame.copy()
    if ground:
        mask = ground_mask((H, W), ground, ring)
        tint = out.copy()
        tint[mask > 0] = (200, 80, 0)
        out = cv2.addWeighted(tint, 0.3, out, 0.7, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 120, 0), 3)
    if ground_crop:
        x, y, w, h = ground_crop
        cv2.rectangle(out, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (0, 230, 255), 3)
    if ring:
        mask = motion_mask((H, W), ring, spots)
        tint = out.copy()
        tint[mask > 0] = (0, 200, 0)
        out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
    if crop:
        x, y, w, h = crop
        cv2.rectangle(out, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (0, 160, 255), 4)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out} ({W}x{H}); ring={ring} crop={crop} ground={ground} ground_crop={ground_crop}")


if __name__ == "__main__":
    main()
