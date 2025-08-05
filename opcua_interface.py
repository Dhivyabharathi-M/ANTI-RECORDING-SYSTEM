from opcua import Server, ua
import time
import numpy as np

class PhoneDetectionOPCUAServer:
    def _init_(self,
                 endpoint="opc.tcp://0.0.0.0:4840",
                 namespace_uri="http://example.org/coligo_phone",
                 thermal_crop_shape=(16, 16)):
        self.server = Server()
        self.server.set_endpoint(endpoint)
        self.server.set_server_name("COLIGOPhoneDetectionServer")

        self.idx = self.server.register_namespace(namespace_uri)
        objects = self.server.get_objects_node()
        self.folder = objects.add_folder(self.idx, "PhoneDetectionData")

        self.phone_detected = self.folder.add_variable(self.idx, "PhoneDetected", False)
        self.phone_confidence = self.folder.add_variable(self.idx, "PhoneConfidence", 0.0)
        self.phone_bbox = self.folder.add_variable(self.idx, "PhoneBBox", [0, 0, 0, 0])  # x,y,w,h

        h, w = thermal_crop_shape  
        initial_crop = np.zeros((h, w), dtype=float).tolist()
        self.phone_temp_array = self.folder.add_variable(
            self.idx, "PhoneTempArray", initial_crop, varianttype=ua.VariantType.Double
        )
        self.phone_temp_array.set_value_rank(2)
        self.phone_temp_array.set_array_dimensions([h, w])

        self.phone_hotspot = self.folder.add_variable(self.idx, "PhoneHotspotTemperature", 0.0)
        self.phone_average = self.folder.add_variable(self.idx, "PhoneAverageTemperature", 0.0)

        self.alarm = self.folder.add_variable(self.idx, "PhoneAlarm", False)
        self.status = self.folder.add_variable(self.idx, "Status", "Initializing")
        self.last_update = self.folder.add_variable(self.idx, "LastUpdateUnix", int(time.time()))

        self.lowlight_gain = self.folder.add_variable(self.idx, "LowLightGain", 2.0)
        self.clahe_clip_limit = self.folder.add_variable(self.idx, "CLAHEClipLimit", 3.0)
        self.detection_threshold = self.folder.add_variable(self.idx, "DetectionThreshold", 0.8)
        self.persistence_required = self.folder.add_variable(self.idx, "PersistenceRequiredFrames", 3)
        self.min_blob_area = self.folder.add_variable(self.idx, "MinBlobArea", 3)
        self.max_blob_area = self.folder.add_variable(self.idx, "MaxBlobArea", 200)

        self.frame_rate = self.folder.add_variable(self.idx, "FrameRate", 0.0)
        self.model_mode = self.folder.add_variable(self.idx, "ModelMode", "Inference")
        self.model_ready = self.folder.add_variable(self.idx, "ModelReady", False)

        for var in [self.lowlight_gain, self.clahe_clip_limit,
                    self.detection_threshold, self.persistence_required,
                    self.min_blob_area, self.max_blob_area]:
            var.set_writable()

        for var in [self.phone_detected, self.phone_confidence, self.phone_bbox,
                    self.phone_temp_array, self.phone_hotspot, self.phone_average,
                    self.alarm, self.status, self.last_update,
                    self.frame_rate, self.model_mode, self.model_ready]:
            pass

    def start(self):
        self.server.start()
        self.status.set_value("Running")
        self.model_ready.set_value(True)

    def stop(self):
        self.server.stop()

    def update_detection(self,
                         detected: bool,
                         confidence: float,
                         bbox: tuple[int, int, int, int],
                         temp_crop: np.ndarray | None,
                         hotspot: float = 0.0,
                         average: float = 0.0,
                         alarm_condition: bool | None = None,
                         model_mode: str = "Inference",
                         frame_rate: float | None = None):
        """
        Push a new detection/inference result into OPC UA variables.

        :param detected: whether phone was detected
        :param confidence: detection confidence (0..1)
        :param bbox: (x, y, w, h)
        :param temp_crop: downsampled grayscale crop (2D numpy array) to map to simulated temperature
        :param hotspot: precomputed hotspot (if you want to supply instead of computing)
        :param average: precomputed average
        :param alarm_condition: override alarm (if None, sets same as detected)
        :param model_mode: "Training" or "Inference"
        :param frame_rate: measured FPS (optional)
        """
        timestamp = int(time.time())
        try:
            self.phone_detected.set_value(bool(detected))
            self.phone_confidence.set_value(float(confidence))
            self.phone_bbox.set_value([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

            if temp_crop is not None:
                arr = temp_crop.astype(float).tolist()
                self.phone_temp_array.set_value(arr)
                if hotspot == 0.0:
                    hotspot = float(np.max(temp_crop))
                if average == 0.0:
                    average = float(np.mean(temp_crop))
            self.phone_hotspot.set_value(float(hotspot))
            self.phone_average.set_value(float(average))

            alarm_val = detected if alarm_condition is None else bool(alarm_condition)
            self.alarm.set_value(alarm_val)

            self.status.set_value("PhoneDetected" if detected else "NoPhone")
            self.last_update.set_value(timestamp)
            self.model_mode.set_value(model_mode)
            if frame_rate is not None:
                self.frame_rate.set_value(float(frame_rate))
        except Exception as e:
            print("Error updating OPC UA variables:", e)