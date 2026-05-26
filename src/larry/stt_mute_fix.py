"""GroqSTTService subclass that honours ``_muted`` (workaround for upstream Pipecat bug)."""

from pipecat.frames.frames import AudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.groq.stt import GroqSTTService


class MutedGroqSTTService(GroqSTTService):
    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        # Workaround for pipecat SegmentedSTTService.process_audio_frame not honoring _muted.
        if self._muted:
            return
        await super().process_audio_frame(frame, direction)
