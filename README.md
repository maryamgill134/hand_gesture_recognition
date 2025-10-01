# Advanced Hand Gesture Recognition

## Overview
Real-time hand gesture recognition using Python, OpenCV, and MediaPipe.
Detects multiple hands, smooths gestures, overlays emojis, logs gestures with timestamps, and optionally saves landmarks for ML training.

## Features
- Multi-hand detection & tracking
- Smooth gesture recognition (deque smoothing)
- Gesture logging with timestamps
- Emoji overlay
- Save landmarks for ML training
- Optional gesture-to-action mapping

## Supported Gestures
- Thumbs Up 👍
- Thumbs Down 👎
- Victory ✌️
- Rock 🤘
- Palm Open ✋
- Fist ✊
- Index Up ☝️
- OK 👌
- Unknown ❓

## Installation
1. Clone repo
2. Create virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
