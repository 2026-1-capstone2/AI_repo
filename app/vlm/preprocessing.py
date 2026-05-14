"""
입력 전처리
- sample_video_frames: 영상 → 균등 샘플링된 프레임 텐서
- build_prompt: 대화 히스토리 + 질문 → Qwen2 conversation prompt
"""
from typing import List


def sample_video_frames(video_path: str, num_frames: int = 32, size: int = 432):
    """영상을 균등 샘플링해 [F, 3, size, size] 텐서 반환.

    decord/torch가 없는 환경(stub)에서도 import 에러 안 나도록 lazy import.
    """
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames).astype(int)
    frames = vr.get_batch(idx).asnumpy()  # [F, H, W, 3]

    resized = []
    for f in frames:
        img = Image.fromarray(f).resize((size, size), Image.BICUBIC)
        resized.append(np.asarray(img))
    arr = np.stack(resized, axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def build_prompt(history: List, question: str) -> str:
    """대화 히스토리 + 새 질문 → Qwen2 prompt.

    이미지 토큰은 전체 대화에서 첫 user 메시지에만 1회 부착.
    """
    from app.vlm.llava.constants import DEFAULT_IMAGE_TOKEN      # type: ignore
    from app.vlm.llava.conversation import conv_templates         # type: ignore

    conv = conv_templates["qwen_1_5"].copy()
    image_token_used = False

    for msg in history:
        role = conv.roles[0] if msg.role == "user" else conv.roles[1]
        if msg.role == "user" and not image_token_used:
            content = DEFAULT_IMAGE_TOKEN + "\n" + msg.content
            image_token_used = True
        else:
            content = msg.content
        conv.append_message(role, content)

    new_user_content = (
        (DEFAULT_IMAGE_TOKEN + "\n" + question) if not image_token_used else question
    )
    conv.append_message(conv.roles[0], new_user_content)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()
