"""Subscribe to Pi camera publications and republish decoded video as local RTSP."""
import asyncio
import logging
import os
import subprocess

import requests
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sfu-subscriber")

PI_API_URL = os.environ["PI_API_URL"].rstrip("/")
ACCESS_CLIENT_ID = os.environ["CF_ACCESS_CLIENT_ID"]
ACCESS_CLIENT_SECRET = os.environ["CF_ACCESS_CLIENT_SECRET"]
RTSP_BASE_URL = os.getenv("RTSP_BASE_URL", "rtsp://rtsp-server:8554")


def signaling(method, path, **kwargs):
    response = requests.request(
        method,
        f"{PI_API_URL}{path}",
        headers={
            "CF-Access-Client-Id": ACCESS_CLIENT_ID,
            "CF-Access-Client-Secret": ACCESS_CLIENT_SECRET,
        },
        timeout=40,
        **kwargs,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


async def main():
    publication = signaling("POST", "/api/sfu/publish")
    publisher_session_id = publication["session_id"]
    subscription = signaling(
        "POST", "/api/sfu/subscribe", json={"publisher_session_id": publisher_session_id}
    )
    session_id = subscription["subscriber_session_id"]
    description = subscription["sessionDescription"]
    track_names = {
        str(track["mid"]): track["trackName"]
        for track in subscription["tracks"]
        if track.get("mid") is not None and track.get("trackName") in ("cam0", "cam1")
    }
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.cloudflare.com:3478"])]))
    writers = {}

    @pc.on("track")
    def on_track(track):
        transceiver = next((item for item in pc.getTransceivers() if item.receiver.track.id == track.id), None)
        mid = transceiver.mid if transceiver else None
        name = track_names.get(str(mid))
        if not name:
            logger.error("Received SFU track without a known camera mapping")
            return
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-video_size", "1280x720", "-framerate", "30",
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-tune", "zerolatency", "-f", "rtsp", "-rtsp_transport", "tcp", f"{RTSP_BASE_URL}/{name}"
        ], stdin=subprocess.PIPE)
        writers[name] = process
        asyncio.create_task(forward_track(track, process, name))

    async def forward_track(track, process, name):
        try:
            while True:
                frame = await track.recv()
                image = frame.to_ndarray(format="bgr24")
                if process.poll() is not None:
                    raise RuntimeError(f"FFmpeg bridge exited for {name}")
                process.stdin.write(image.tobytes())
        except Exception:
            logger.exception("Video forwarding stopped for %s", name)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(**description))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice(pc)
        signaling(
            "PUT", f"/api/sfu/subscribe/{session_id}/answer",
            json={"sessionDescription": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}},
        )
        await wait_for_connection(pc)
        logger.info("Subscribed; tracks publishing to %s/cam0 and %s/cam1", RTSP_BASE_URL, RTSP_BASE_URL)
        await asyncio.Future()
    finally:
        await pc.close()
        try:
            signaling("DELETE", f"/api/sfu/subscribe/{session_id}")
        except Exception:
            logger.exception("Failed to clean up SFU subscriber session")
        try:
            signaling("DELETE", "/api/sfu/publish")
        except Exception:
            logger.exception("Failed to stop Pi SFU publisher")
        for process in writers.values():
            if process.stdin:
                process.stdin.close()
            process.terminate()


async def wait_for_connection(pc, timeout=30):
    if pc.connectionState == "connected":
        return
    event = asyncio.Event()

    @pc.on("connectionstatechange")
    async def on_connection_state():
        if pc.connectionState in ("connected", "failed", "closed"):
            event.set()

    await asyncio.wait_for(event.wait(), timeout)
    if pc.connectionState != "connected":
        raise RuntimeError(f"SFU subscriber connection ended in state {pc.connectionState}")


async def wait_for_ice(pc, timeout=25):
    if pc.iceGatheringState == "complete":
        return
    event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def on_ice_state():
        if pc.iceGatheringState == "complete":
            event.set()

    await asyncio.wait_for(event.wait(), timeout)


if __name__ == "__main__":
    asyncio.run(main())
