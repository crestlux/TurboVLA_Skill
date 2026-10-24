import atexit
import copy
import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import pyrealsense2 as rs


class RSCapture:
    def __init__(
        self,
        name,
        serial_number,
        dim=(1920, 1080),
        fps=30,
        depth=False,
        exposure=None,
        queue_size=10,
    ):
        self.name = name
        self.depth = depth
        self.dim = dim
        self.stopped = False
        self.frame_queue = Queue(maxsize=queue_size)  # 큐 생성
        self.__queue_rgb_buffer = np.zeros(
            (queue_size, dim[1], dim[0], 3), dtype=np.uint8
        )
        if self.depth:
            self.__queue_depth_buffer = np.zeros(
                (queue_size, dim[1], dim[0], 1), dtype=np.uint16
            )
        else:
            self.__queue_depth_buffer = [
                None,
            ] * queue_size
        self.__queue_buffer_idx = 0

        self.__last_rgb = np.zeros((dim[1], dim[0], 3), dtype=np.uint8)
        self.__last_timestamp = 0.0

        # Initialize the pipeline and configure streams
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        try:
            atexit.register(self.try_close)
            assert serial_number in self.get_device_serial_numbers(), (
                "RS device not found"
            )
            self.serial_number = serial_number
            self.cfg.enable_device(self.serial_number)
            self.cfg.enable_stream(
                rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps
            )
            if self.depth:
                self.cfg.enable_stream(
                    rs.stream.depth, dim[0], dim[1], rs.format.z16, fps
                )
            self.profile = self.pipe.start(self.cfg)
            
            # if exposure is not None:
            #     device = self.profile.get_device()

            #     color_sensor = device.first_color_sensor()

            #     if color_sensor is None:
            #         raise RuntimeError("Color sensor not found")

            #     self._sensor = color_sensor
                
            #     self._sensor.set_option(rs.option.enable_auto_exposure, 0)

            #     # 2. Exposure 설정 (μs 단위)
            #     self._sensor.set_option(rs.option.exposure, exposure)
            # else:
            # Set camera exposure
            # self._sensor = self.profile.get_device().query_sensors()[0]
            device = self.profile.get_device()
            if "D405" in device.get_info(rs.camera_info.name):
                self._sensor = device.first_depth_sensor()
            else:
                self._sensor = device.first_color_sensor()
            # exposure maunal 설정
            # self._sensor.set_option(rs.option.exposure, exposure)
            # exposure auto 설정
            if exposure is None:
                self._sensor.set_option(rs.option.enable_auto_exposure, 1)
            else:
                self._sensor.set_option(rs.option.enable_auto_exposure, 0)
                self._sensor.set_option(rs.option.exposure, float(exposure))

            # Align depth to color stream
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # Ensure resources are released on exit
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            self.close()
            raise

    @staticmethod
    def get_device_serial_numbers():
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def get_timestamp(self, frames):
        timestamps = {
            "global_timestamp": None,
            "backend_timestamp": frames.get_frame_metadata(
                rs.frame_metadata_value.backend_timestamp
            )
            / 1000,
        }

        try:
            sensor_timestamp = (
                frames.get_frame_metadata(
                    rs.frame_metadata_value.sensor_timestamp
                )
                / 1000
            )
            frame_timestamp = (
                frames.get_frame_metadata(
                    rs.frame_metadata_value.frame_timestamp
                )
                / 1000
            )
            timestamps["global_timestamp"] = timestamps[
                "backend_timestamp"
            ] - (frame_timestamp - sensor_timestamp)
            timestamps["sensor_timestamp"] = sensor_timestamp
            timestamps["frame_timestamp"] = frame_timestamp
        except RuntimeError:
            # Fallback to backend timestamp if others are unavailable
            timestamps["global_timestamp"] = timestamps["backend_timestamp"]

        return timestamps

    def start(self):
        # Start the frame update thread
        self.stopped = False
        self._update_thread = Thread(target=self.update, daemon=True)
        self._update_thread.start()
        return self

    def connect(self):
        self.start()  # alias

    def update(self):
        while not self.stopped:
            try:
                frames = self.pipe.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                if not color_frame.is_video_frame():
                    continue

                # Process color frame
                rgb_buf = np.frombuffer(
                    color_frame.get_data(), dtype=np.uint8
                ).reshape((
                    self.dim[1],
                    self.dim[0],
                    3,
                ))  # reshape is O(1) - almost equal to a tuple assignment to variables
                self.__queue_rgb_buffer[self.__queue_buffer_idx] = rgb_buf

                if self.depth:
                    depth_frame = aligned_frames.get_depth_frame()
                    if depth_frame.is_depth_frame():
                        depth_buf = np.frombuffer(
                            depth_frame.get_data(), dtype=np.uint16
                        ).reshape((self.dim[1], self.dim[0], 1))
                        self.__queue_depth_buffer[self.__queue_buffer_idx] = (
                            depth_buf
                        )

                # Prepare frame data
                frame_data = {
                    "rgb_image": self.__queue_rgb_buffer[
                        self.__queue_buffer_idx
                    ],
                    "depth_image": self.__queue_depth_buffer[
                        self.__queue_buffer_idx
                    ],
                    "timestamp": self.get_timestamp(frames),
                }
                self.__queue_buffer_idx = (
                    self.__queue_buffer_idx + 1
                ) % self.frame_queue.maxsize
                self.__last_rgb[:] = frame_data["rgb_image"]
                self.__last_timestamp = (
                    time.time()
                )  # frame_data["timestamp"]["backend_timestamp"]

                # Add frame to queue (discard oldest if full)
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame_data)

            except Exception as e:
                print(f"Error capturing frame: {e}")
                self.stopped = True

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Return the latest rgb image read.
        """
        if self.stopped:
            raise RuntimeError("Camera is stopped")

        return self.__last_timestamp != 0, self.__last_rgb

    def get_closest_frame(self, target_timestamp):
        closest_frame = None
        closest_diff = float("inf")  # 무한대로 초기화

        with (
            self.frame_queue.mutex
        ):  # queue.Queue의 내부 데이터를 안전하게 접근
            frames = list(
                self.frame_queue.queue
            )  # 현재 큐에 있는 모든 데이터를 가져옴
        for frame in frames:
            frame_timestamp = frame["timestamp"]["backend_timestamp"]
            diff = abs(frame_timestamp - target_timestamp)
            if diff < closest_diff:
                closest_diff = diff
                closest_frame = frame

        return closest_frame

    def close(self):
        if self.stopped:
            return
        # Tell the background update() thread to stop calling
        # pipe.wait_for_frames() and wait for it to actually exit before we
        # touch the pipe, or it can race pipe.stop() and throw.
        self.stopped = True
        update_thread = getattr(self, "_update_thread", None)
        if update_thread is not None and update_thread.is_alive():
            update_thread.join(timeout=2.0)
        if hasattr(self, "pipe"):
            self.pipe.stop()
        if hasattr(self, "cfg"):
            self.cfg.disable_all_streams()

    def try_close(self):  # for silent closing
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        self.try_close()

    def show(
        self, image=None, show_dim: tuple[int, int] = None, process_fn=None
    ):
        if image is None:
            self.read()
            image = self.__last_rgb
        if process_fn is not None:
            image = process_fn(image)
        if show_dim is not None:
            image = cv2.resize(image, show_dim)
        # print(image.shape, type(image))
        if image is not None:
            cv2.imshow(f"RSCamera - {self.name}", image)
            cv2.waitKey(1)

    @property
    def last_timestamp(self):
        return self.__last_timestamp


if __name__ == "__main__":
    import copy
    import time

    import cv2

    CAMERAS = {
        "head_cam": {
            "serial_number": "213522254901",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }

    cam = RSCapture("meh", **CAMERAS["head_cam"])
    try:
        cam.start()

        for _ in range(100000):
            ret, frame = cam.read()
            if ret:
                # cv2.resize(frame[...], (512, 512))
                cv2.imshow(
                    "test",
                    frame[48: 48+256, 450 : 450 + 256],
                    # 742, 718
                    # 738, 1080
                    # np.concatenate([
                    # cv2.resize(
                    #    frame, (512, 512)
                    # ),
                    # cv2.resize(
                    #     frame[738-64:738+64, 1080-64:1080+64], (512, 512)
                    # ),
                    # ], axis=1)
                )
            cv2.waitKey(33)
            time.sleep(0.034)
    finally:
        try:
            cam.try_close()
            cv2.destroyAllWindows()
        finally:
            pass
