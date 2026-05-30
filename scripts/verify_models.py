"""
GPU 서버 실모델 검증 스크립트 (TODO.md §C-VERIFY V1~V5)

S3 / RabbitMQ 인프라 없이 **모델 경로만** 직접 호출하여 실모델이 GPU 서버에서
정상 동작하는지 확인한다. 각 단계는 TODO.md 의 V1~V5 체크리스트에 대응한다.

사전 조건:
  - STUB_MODELS=false
  - 모델 가중치 다운로드 완료 (README §2.4)
      CUT3R   : settings.cut3r_weights
      LoRA    : settings.model_path
      베이스   : HF 캐시 (최초 실행 시 자동 다운로드, ~14GB)

사용법:
  python -m scripts.verify_models                 # 전체(V1~V5) — sample.mp4 자동 생성
  python -m scripts.verify_models --video a.mp4   # 실제 영상으로 검증
  python -m scripts.verify_models --only 1,2,3    # 일부 단계만 (V5 14GB 다운로드 회피)
  python -m scripts.verify_models --skip 5        # V5 제외

출력 말미에 TODO.md "검증 결과 회신 양식" 형태로 PASS/FAIL 요약을 찍는다.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

# 검증 단계 기대값
EXPECT_CAMERA = (32, 1, 768)
EXPECT_PATCH = (32, 729, 768)
SAMPLE_DEFAULT = "/tmp/verify_sample.mp4"


def _make_sample_video(path: str) -> str:
    """테스트용 더미 mp4 생성 (5초, 10fps, 640x480)."""
    import numpy as np
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
    for i in range(50):
        frame = np.full((480, 640, 3), (i * 5) % 255, dtype=np.uint8)
        cv2.rectangle(frame, (50 + i, 50), (200 + i, 200), (0, 255, 0), -1)
        vw.write(frame)
    vw.release()
    return path


# ──────────────────────────────────────────────────────────────────
# 각 검증 단계. cut3r/vlm 핸들을 캐시해 중복 로드를 피한다.
# ──────────────────────────────────────────────────────────────────
class _State:
    cut3r = None
    vlm = None


def v1_load_cut3r() -> str:
    """V1: CUT3R 가중치 정상 로드 — class/device/dtype, VRAM 증가."""
    import torch
    from app.vlm.loader import load_cut3r

    before = torch.cuda.memory_allocated() / 1e9
    _State.cut3r = load_cut3r()
    after = torch.cuda.memory_allocated() / 1e9
    h = _State.cut3r
    cls = type(h["model"]).__name__
    assert cls == "Cut3rEncoder", f"expected Cut3rEncoder, got {cls}"
    assert h["device"] == "cuda", f"expected cuda, got {h['device']}"
    assert h["dtype"] is torch.float16, f"expected float16, got {h['dtype']}"
    return f"{cls} {h['device']} {h['dtype']}, VRAM +{after - before:.2f}GB"


def v2_decord_compat(video: str) -> str:
    """V2: process_video_with_decord 가 로컬 _VideoArgs(3필드) 와 호환."""
    from app.vlm.llava.utils import process_video_with_decord
    from app.vlm.cut3r_runner import _VideoArgs

    frames, vt, ft, n = process_video_with_decord(video, _VideoArgs())
    f0 = frames[0]
    shape = getattr(f0, "shape", None)
    assert len(frames) > 0, "no frames sampled"
    assert shape is not None, f"frame0 has no shape (type={type(f0).__name__})"
    return f"{len(frames)} frames, frame0={type(f0).__name__}{tuple(shape)}, video_time={vt}"


def v3_forward_signature() -> str:
    """V3: Cut3rEncoder(x, point_cloud_output_paths=None) forward 시그니처."""
    import torch

    h = _State.cut3r
    assert h is not None, "V1 must run before V3"
    x = torch.randn(32, 3, 432, 432, device=h["device"], dtype=h["dtype"])
    with torch.inference_mode():
        cam, patch = h["model"](x, point_cloud_output_paths=None)
    assert tuple(cam.shape) == EXPECT_CAMERA, f"camera {tuple(cam.shape)} != {EXPECT_CAMERA}"
    assert tuple(patch.shape) == EXPECT_PATCH, f"patch {tuple(patch.shape)} != {EXPECT_PATCH}"
    return f"camera={tuple(cam.shape)} patch={tuple(patch.shape)} {cam.dtype}"


def v4_extract_and_serialize(video: str) -> str:
    """V4: 실영상 → extract_spatial_features → .pt 저장/재로드 (S3 우회)."""
    import os
    import torch
    from app.vlm.cut3r_runner import extract_spatial_features

    h = _State.cut3r
    assert h is not None, "V1 must run before V4"
    t0 = time.time()
    feats = extract_spatial_features(h, video)
    dt = (time.time() - t0) * 1000
    cam, patch = feats["camera_tokens"], feats["patch_tokens"]
    assert tuple(cam.shape) == EXPECT_CAMERA, f"camera {tuple(cam.shape)} != {EXPECT_CAMERA}"
    assert tuple(patch.shape) == EXPECT_PATCH, f"patch {tuple(patch.shape)} != {EXPECT_PATCH}"
    assert str(cam.device) == "cpu" and cam.dtype is torch.float16, "expected cpu/fp16 for serialization"

    out = "/tmp/verify_spatial.pt"
    torch.save(feats, out)
    re = torch.load(out, map_location="cpu")
    assert tuple(re["camera_tokens"].shape) == EXPECT_CAMERA
    size_kb = os.path.getsize(out) // 1024
    return f"extract {dt:.0f}ms, .pt {size_kb}KB, reload OK"


def v5_real_inference(video: str) -> str:
    """V5: load_vlm + model.generate 실모델 추론 (S3/RabbitMQ 우회)."""
    import torch
    from app.vlm.loader import load_vlm, load_cut3r
    from app.vlm.cut3r_runner import extract_spatial_features
    from app.vlm.preprocessing import sample_video_frames, build_prompt
    from app.vlm.llava.constants import IMAGE_TOKEN_INDEX
    from app.vlm.llava.mm_utils import tokenizer_image_token

    if _State.vlm is None:
        t0 = time.time()
        _State.vlm = load_vlm()
        print(f"  [V5] load_vlm OK in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    if _State.cut3r is None:
        _State.cut3r = load_cut3r()

    tokenizer = _State.vlm["tokenizer"]
    model = _State.vlm["model"]
    image_processor = _State.vlm["image_processor"]

    feats = extract_spatial_features(_State.cut3r, video)
    spatial_features = [{
        "camera_tokens": feats["camera_tokens"].to("cuda", dtype=torch.float16),
        "patch_tokens": feats["patch_tokens"].to("cuda", dtype=torch.float16),
    }]
    frames = sample_video_frames(video, num_frames=32, size=432)
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)

    prompt = build_prompt([], "What objects are in this room?")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=[video_tensor],  # video modality는 리스트로 전달 (inference.py 와 동일)
            attention_mask=attention_mask,
            spatial_features=spatial_features,
            modalities="video",
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128,
            use_cache=True,
        )
    dt = (time.time() - t0) * 1000
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    assert dt > 0 and len(answer) > 0, "empty inference result"
    return f"{dt:.0f}ms, {output_ids.shape[1]} tok, answer={answer[:80]!r}"


STEPS = {
    1: ("V1 CUT3R 로드", lambda a: v1_load_cut3r()),
    2: ("V2 decord 호환", lambda a: v2_decord_compat(a.video)),
    3: ("V3 forward 시그니처", lambda a: v3_forward_signature()),
    4: ("V4 추출+직렬화", lambda a: v4_extract_and_serialize(a.video)),
    5: ("V5 실모델 추론", lambda a: v5_real_inference(a.video)),
}


def main():
    ap = argparse.ArgumentParser(description="GPU 서버 실모델 검증 (C-VERIFY V1~V5)")
    ap.add_argument("--video", default=None, help="검증용 영상 경로 (미지정 시 더미 생성)")
    ap.add_argument("--only", default=None, help="실행할 단계 (예: 1,2,3)")
    ap.add_argument("--skip", default=None, help="제외할 단계 (예: 5)")
    args = ap.parse_args()

    # 실모델 모드 강제 확인
    from app.core.config import settings
    if settings.stub_models:
        print("ERROR: STUB_MODELS=true 입니다. 실모델 검증은 STUB_MODELS=false 필요.", file=sys.stderr)
        sys.exit(2)

    # 검증 영상 준비
    if not args.video:
        args.video = _make_sample_video(SAMPLE_DEFAULT)
        print(f"[setup] 더미 검증 영상 생성: {args.video}", flush=True)

    only = {int(x) for x in args.only.split(",")} if args.only else set(STEPS)
    skip = {int(x) for x in args.skip.split(",")} if args.skip else set()
    selected = [n for n in sorted(STEPS) if n in only and n not in skip]

    results: dict[int, tuple[str, str]] = {}
    for n in selected:
        title, fn = STEPS[n]
        print(f"\n=== {title} ===", flush=True)
        try:
            detail = fn(args)
            results[n] = ("PASS", detail)
            print(f"  PASS — {detail}", flush=True)
        except Exception as e:
            results[n] = ("FAIL", f"{type(e).__name__}: {e}")
            print(f"  FAIL — {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    # TODO.md 회신 양식
    print("\n" + "=" * 50, flush=True)
    print("검증 결과 회신 양식", flush=True)
    print("=" * 50, flush=True)
    for n in sorted(STEPS):
        if n not in results:
            print(f"V{n} 결과: [ SKIP ]", flush=True)
            continue
        verdict, detail = results[n]
        print(f"V{n} 결과: [ {verdict} ] — {detail}", flush=True)

    if any(v == "FAIL" for v, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
