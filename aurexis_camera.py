"""
aurexis_camera.py — live camera inference using the substrate.

Captures from your webcam, computes atoms on every frame, applies the
learned predicate library, overlays results live on the video.

This is the substrate's deployment endpoint. Unlike a CNN that outputs
"cat: 0.87" with no reasoning, the substrate says:
    edge_density=0.182 (HIGH)
    color_diversity=2.4 (LOW)
    matches: cartographic_texture, k7c2_high_contrast
Every classification has a complete paper trail.

Controls during live mode:
    q         quit
    s         save current frame's measurements as a teach example
    space     pause / resume
    +  /  -   change which top-N predicates to display
    1..9      toggle individual predicates by index
"""

import time
import numpy as np
from PIL import Image
from collections import deque

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def live_camera(archive, camera_index=0, target_fps=15,
                show_top_n=8, frame_size=(640, 480)):
    """Open the webcam and run live substrate inference.

    Args:
        archive       aurexis_v3.Archive (must have predicates in library)
        camera_index  webcam device id (0 = default)
        target_fps    target frame rate
        show_top_n    how many predicates to display per frame
        frame_size    (width, height) capture resolution
    """
    if not HAS_CV2:
        print("\n  [camera] OpenCV not installed.")
        print("           Run: pip install opencv-python")
        return

    if not archive.library:
        print("\n  [camera] Predicate library is empty.")
        print("           Run [r] (auto-discover rules) first to populate it.")
        return

    from aurexis_v3 import ATOMS
    atom_list = list(ATOMS.values())

    print(f"\n  opening camera index {camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [camera] could not open device {camera_index}")
        print(f"           try a different index (1, 2, etc.)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    cap.set(cv2.CAP_PROP_FPS,           target_fps)

    # Pre-build predicate list
    predicates = list(archive.library.items())
    print(f"  loaded {len(predicates)} predicates from library")
    print(f"  press q=quit, s=save, space=pause, +/-=display count\n")

    paused = False
    last_atoms = {}
    last_matches = []
    fps_history = deque(maxlen=30)
    saved_count = 0

    try:
        while True:
            t_frame = time.time()
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("  [camera] frame read failed; stopping")
                    break

                # Convert BGR (OpenCV) -> RGB numpy for atom computation
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize for fast atom computation; camera frames are big
                small = cv2.resize(rgb, (256, 256))

                # Compute atoms on small frame
                atoms = {a.name: float(a(small)) for a in atom_list}
                last_atoms = atoms

                # Apply each predicate
                # Threshold predicates (rule-based) only need atom values
                # Learned predicates need rich features which are too slow
                # for real-time, so we skip them in live mode
                matches = []
                for name, pred in predicates:
                    if pred.kind == "rule":
                        try:
                            if pred.matches(atoms, features=None):
                                matches.append(name)
                        except Exception:
                            continue
                last_matches = matches

            # Build overlay text
            overlay_lines = []
            overlay_lines.append(
                f"AUREXIS LIVE  |  predicates: {len(last_matches)}/"
                f"{len(predicates)} matching"
            )
            if fps_history:
                overlay_lines.append(
                    f"FPS: {1.0/np.mean(fps_history):.1f}"
                )

            # Show top atoms (ones with most variance from typical)
            overlay_lines.append("")
            overlay_lines.append("--- atoms ---")
            for name in ["mean_brightness", "contrast", "edge_density",
                         "color_diversity", "lbp_uniformity"]:
                if name in last_atoms:
                    overlay_lines.append(f"  {name}: {last_atoms[name]:.3f}")

            # Show top-N matching predicates
            overlay_lines.append("")
            overlay_lines.append(f"--- matches (top {show_top_n}) ---")
            for name in last_matches[:show_top_n]:
                overlay_lines.append(f"  {name[:50]}")
            if len(last_matches) > show_top_n:
                overlay_lines.append(
                    f"  ... +{len(last_matches) - show_top_n} more"
                )

            # Render text on frame
            y = 24
            for line in overlay_lines:
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if line.startswith("--") or
                            "AUREXIS" in line else (255, 255, 255),
                            1, cv2.LINE_AA)
                y += 18

            # PAUSED indicator
            if paused:
                cv2.putText(frame, "[ PAUSED ]",
                            (frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Aurexis Live", frame)

            # Track FPS
            dt = time.time() - t_frame
            fps_history.append(dt)

            # Key handling
            key = cv2.waitKey(max(1, int(1000 / target_fps))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"  [{'paused' if paused else 'running'}]")
            elif key == ord('s'):
                # Save the current frame's data as an archive entry
                if not paused and last_atoms:
                    pil = Image.fromarray(rgb)
                    alias = f"camera_{int(time.time())}_{saved_count:03d}"
                    h, is_new = archive.ingest_pil(pil, alias, "camera")
                    saved_count += 1
                    print(f"  saved: {alias} ({'NEW' if is_new else 'dup'})")
                    print(f"    matched {len(last_matches)} predicates")
            elif key == ord('+') or key == ord('='):
                show_top_n = min(show_top_n + 2, 30)
                print(f"  show_top_n = {show_top_n}")
            elif key == ord('-'):
                show_top_n = max(show_top_n - 2, 2)
                print(f"  show_top_n = {show_top_n}")

    except KeyboardInterrupt:
        print("\n  interrupted")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  camera closed. saved {saved_count} frames.")
