"""Run both YOLO pipelines directly from Cloudflare SFU WebRTC tracks."""
import asyncio
import logging
import os
import queue
import threading
import time

import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
import requests

from pipeline_runner import MLPipeline, load_config_from_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct-frame-runner")


class FrameSource:
    """OpenCV-compatible, bounded latest-frame source for one camera track."""
    def __init__(self, width=1280, height=720):
        self.frames = queue.Queue(maxsize=1)
        self.width, self.height = width, height
        self.closed = False

    def push(self, frame):
        if self.closed:
            return
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)

    def read(self):
        while not self.closed:
            try:
                return True, self.frames.get(timeout=1)
            except queue.Empty:
                continue
        return False, None

    def isOpened(self):
        return not self.closed

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cv2.CAP_PROP_FPS:
            return 25
        return 0

    def release(self):
        self.closed = True


def signaling(method, path, **kwargs):
    response = requests.request(
        method, f"{os.environ['PI_API_URL'].rstrip('/')}{path}",
        headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                 "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]},
        timeout=40, **kwargs)
    if not response.ok:
        error = requests.HTTPError(f"{response.status_code} {response.reason}: {response.text[:500]}")
        error.response = response
        raise error
    return response.json() if response.content else {}

async def create_subscription_with_retry(publisher_id, attempts=20, delay=2):
    for attempt in range(1, attempts + 1):
        try:
            return signaling("POST", "/api/sfu/subscribe", json={"publisher_session_id": publisher_id})
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = exc.response.text if exc.response is not None else str(exc)
            if status != 502 or attempt == attempts:
                raise
            logger.warning("SFU subscription not ready (attempt %d/%d, HTTP %s): %s",
                           attempt, attempts, status, body[:300])
            await asyncio.sleep(delay)

async def wait_for_ice(pc, timeout=20):
    if pc.iceGatheringState == "complete":
        return
    event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def on_ice():
        if pc.iceGatheringState == "complete":
            event.set()

    await asyncio.wait_for(event.wait(), timeout)


def log_peer_connection(pc, label):
    @pc.on("icegatheringstatechange")
    async def on_ice_gathering_state():
        logger.info("SFU %s ICE gathering state: %s", label, pc.iceGatheringState)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state():
        logger.info("SFU %s ICE connection state: %s", label, pc.iceConnectionState)

    @pc.on("connectionstatechange")
    async def on_connection_state():
        logger.info("SFU %s peer connection state: %s", label, pc.connectionState)

async def wait_for_connection(pc, timeout=30):
    logger.info("SFU subscriber waiting for connection (ICE=%s, peer=%s)", pc.iceConnectionState, pc.connectionState)
    if pc.connectionState == "connected":
        return
    event = asyncio.Event()

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("connected", "failed", "closed"):
            event.set()

    try:
        await asyncio.wait_for(event.wait(), timeout)
    except asyncio.TimeoutError:
        logger.error("SFU subscriber connection timed out (ICE=%s, peer=%s)", pc.iceConnectionState, pc.connectionState)
        raise
    if pc.connectionState != "connected":
        raise RuntimeError(f"SFU subscriber ended in state {pc.connectionState} (ICE={pc.iceConnectionState})")


async def run():
    sources = {name: FrameSource() for name in ("cam0", "cam1")}
    pipelines = []
    threads = []
    pc = None
    publisher_id = subscriber_id = None
    try:
        base = load_config_from_env()
        for index, name in enumerate(("cam0", "cam1")):
            config = type(base)(**{**base.__dict__, "pipeline_id": f"primary-stream-{index + 1}",
                "pipeline_type": "primary", "stream_url": f"webrtc://{name}",
                "output_stream_url": os.getenv(f"STREAM_{index + 1}_OUTPUT_URL")})
            pipeline = MLPipeline(config, frame_source=sources[name])
            pipelines.append(pipeline)
            thread = threading.Thread(target=pipeline.run, name=f"ml-{name}", daemon=True)
            threads.append(thread)
            thread.start()

        publication = signaling("POST", "/api/sfu/publish")
        publisher_id = publication["session_id"]
        subscription = await create_subscription_with_retry(publisher_id)
        subscriber_id = subscription["subscriber_session_id"]
        track_names = {str(track["mid"]): track["trackName"] for track in subscription.get("tracks", [])
                       if track.get("mid") is not None and track.get("trackName") in sources}
        ice_servers = [RTCIceServer(urls=["stun:stun.cloudflare.com:3478"])]
        turn_urls = [url.strip() for url in os.getenv("CF_SFU_TURN_URLS", "").split(",") if url.strip()]
        turn_username = os.getenv("CF_SFU_TURN_USERNAME", "").strip()
        turn_credential = os.getenv("CF_SFU_TURN_CREDENTIAL", "").strip()
        if turn_urls:
            if not turn_username or not turn_credential:
                raise RuntimeError("CF_SFU_TURN_USERNAME and CF_SFU_TURN_CREDENTIAL are required when CF_SFU_TURN_URLS is set")
            ice_servers.append(RTCIceServer(urls=turn_urls, username=turn_username, credential=turn_credential))
            logger.info("SFU subscriber configured with TURN relay (%d URL(s))", len(turn_urls))
        else:
            logger.warning("SFU subscriber has no TURN server configured; using STUN only")
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        log_peer_connection(pc, "subscriber")

        @pc.on("track")
        def on_track(track):
            transceiver = next((item for item in pc.getTransceivers() if item.receiver.track.id == track.id), None)
            name = track_names.get(str(transceiver.mid)) if transceiver else None
            logger.info("SFU subscriber received track id=%s mid=%s camera=%s", track.id,
                        transceiver.mid if transceiver else None, name)
            if name is None:
                logger.error("Received SFU track without a known camera mapping")
                return
            async def consume():
                frame_count = 0
                try:
                    while True:
                        video_frame = await track.recv()
                        sources[name].push(video_frame.to_ndarray(format="bgr24"))
                        frame_count += 1
                        if frame_count == 1:
                            logger.info("First SFU video frame received for %s", name)
                except MediaStreamError:
                    logger.warning("SFU video track ended for %s after %d frame(s)", name, frame_count)
                except Exception:
                    logger.exception("SFU video track consumer failed for %s", name)
                    raise
            asyncio.create_task(consume())

        logger.info("SFU subscriber remote SDP received; tracks advertised: %s", track_names)
        await pc.setRemoteDescription(RTCSessionDescription(**subscription["sessionDescription"]))
        await pc.setLocalDescription(await pc.createAnswer())
        logger.info("SFU subscriber local SDP answer created; ICE gathering=%s", pc.iceGatheringState)
        await wait_for_ice(pc)
        logger.info("SFU subscriber ICE gathering complete")
        signaling("PUT", f"/api/sfu/subscribe/{subscriber_id}/answer",
                  json={"sessionDescription": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}})
        await wait_for_connection(pc)
        logger.info("Direct SFU frames feeding YOLO cam0 and cam1; no RTSP input bridge")
        while True:
            await asyncio.sleep(5)
            for name, thread in zip(("cam0", "cam1"), threads):
                if not thread.is_alive():
                    raise RuntimeError(
                        f"ML pipeline thread for {name} exited; check preceding pipeline error"
                    )
    finally:
        for source in sources.values():
            source.release()
        for pipeline in pipelines:
            pipeline.running = False
        for thread in threads:
            thread.join(timeout=5)
        if pc:
            await pc.close()
        if subscriber_id:
            try:
                signaling("DELETE", f"/api/sfu/subscribe/{subscriber_id}")
            except Exception:
                logger.exception("Failed to close subscriber session")
        if publisher_id:
            try:
                signaling("DELETE", "/api/sfu/publish")
            except Exception:
                logger.exception("Failed to stop publisher")


if __name__ == "__main__":
    asyncio.run(run())
