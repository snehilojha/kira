"""One-time face enrollment for Kira presence detection.

Usage:
    python -m bot.enroll_face           # enroll
    python -m bot.enroll_face --test    # verify after enrollment

Grabs 5 webcam frames, extracts your face embedding via insightface,
averages them, and saves to data/face_embedding.npy.

Run once before starting Kira. Re-run any time to update your enrollment
(e.g. after a haircut, glasses change, lighting difference).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _get_face_analysis():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def enroll(n_frames: int = 5, output: str = "data/face_embedding.npy") -> None:
    print("Kira face enrollment")
    print("━" * 40)
    print("Loading face model...")
    app = _get_face_analysis()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preview_dir = Path("data/enrollment_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    embeddings: list[np.ndarray] = []
    attempt = 0

    print(f"\nLook at the camera. Capturing {n_frames} frames...")
    print("Press 'q' to quit at any time.\n")

    print("(No preview window — headless mode. Just look at the camera.)\n")

    while len(embeddings) < n_frames:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read frame.")
            break

        attempt += 1
        # Show a dot every frame so user sees it's scanning
        print(f"\r  Scanning... frame {attempt:>4}  |  faces captured: {len(embeddings)}/{n_frames}", end="", flush=True)

        if attempt % 5 != 0:
            continue

        faces = app.get(frame)
        if not faces:
            print(f"\r  frame {attempt:>4} — no face detected, keep looking at camera        ", end="", flush=True)
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(face.normed_embedding)

        # Draw bounding box on a copy and save so user can verify
        preview = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (100, 220, 100), 2)
        cv2.putText(preview, f"face {len(embeddings)}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2)
        preview_path = preview_dir / f"capture_{len(embeddings)}.jpg"
        cv2.imwrite(str(preview_path), preview)

        print(f"\r  ✓ Captured {len(embeddings)}/{n_frames} — saved to {preview_path}          ")
        time.sleep(0.3)

    cap.release()

    if not embeddings:
        print("\nERROR: No face embeddings captured. Enrollment failed.")
        sys.exit(1)

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    np.save(str(out_path), mean_embedding)

    print(f"\nEnrollment complete. Embedding saved to: {out_path}")
    print(f"Captured {len(embeddings)} frames, averaged into one embedding.")
    print("You can now start Kira normally.")


def test(embedding_path: str = "data/face_embedding.npy", n_frames: int = 3) -> None:
    """Grab live frames and print similarity scores against the enrolled embedding."""
    print("Kira face recognition self-test")
    print("━" * 40)

    enrolled_path = Path(embedding_path)
    if not enrolled_path.exists():
        print(f"ERROR: No enrollment found at {enrolled_path}. Run without --test first.")
        sys.exit(1)

    enrolled = np.load(str(enrolled_path))
    print(f"Enrolled embedding loaded from {enrolled_path}")

    print("Loading face model...")
    app = _get_face_analysis()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    THRESHOLD = 0.35
    print(f"\nLook at the camera. Testing {n_frames} frames (threshold = {THRESHOLD})...\n")

    results = []
    attempt = 0
    tested = 0

    while tested < n_frames:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read frame.")
            break

        attempt += 1
        print(f"\r  Scanning... frame {attempt:>4}", end="", flush=True)

        if attempt % 8 != 0:
            continue

        faces = app.get(frame)
        if not faces:
            print(f"\r  frame {attempt:>4} — no face detected                          ", end="", flush=True)
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        similarity = float(np.dot(face.normed_embedding, enrolled))
        passed = similarity >= THRESHOLD
        results.append((similarity, passed))
        tested += 1

        # Save annotated preview
        preview = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        color = (100, 220, 100) if passed else (60, 60, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"sim={similarity:.3f} {'PASS' if passed else 'FAIL'}"
        cv2.putText(preview, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        preview_dir = Path("data/enrollment_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"test_{tested}.jpg"
        cv2.imwrite(str(preview_path), preview)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\r  [{tested}/{n_frames}] similarity = {similarity:.4f}  {status}  → saved to {preview_path}")
        time.sleep(0.4)

    cap.release()

    if not results:
        print("\nNo faces detected during test. Check lighting and camera position.")
        sys.exit(1)

    print("\n" + "━" * 40)
    avg = sum(s for s, _ in results) / len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"Results: {passed_count}/{len(results)} frames passed")
    print(f"Average similarity: {avg:.4f}  (threshold: {THRESHOLD})")

    if avg >= THRESHOLD + 0.10:
        print("Recognition looks solid. You're good to go.")
    elif avg >= THRESHOLD:
        print("Recognition is marginal — consider re-enrolling in better lighting.")
    else:
        print("Recognition failed. Re-enroll with better lighting or closer to camera.")
        print("Tip: run  python -m bot.enroll_face  again to re-enroll.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    else:
        enroll()
