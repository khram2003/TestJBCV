import cv2
import mediapipe as mp
import pathlib
from argparse import ArgumentParser


class MyFaceDetector:
    def __init__(self, path):
        self._input_path = pathlib.Path(path)
        self._cap = cv2.VideoCapture(path)
        frame_width = int(self._cap.get(3))
        frame_height = int(self._cap.get(4))
        self._result = cv2.VideoWriter(self._input_path.name.split('.')[0] + '_processed.avi',
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                       (frame_width, frame_height))
        mpFaceDetection = mp.solutions.face_detection
        self._faceDetection = mpFaceDetection.FaceDetection(0.5)
        self._output_text_file = self._input_path.name.split('.')[0] + '_processed.txt'
        self._detected_moments = {}

    def write_result_to_file(self):
        with open(self._output_text_file, 'w') as f:
            for second, idx in self._detected_moments:
                x, y, _, _ = self._detected_moments[second, idx]
                f.write("time: %d:%d    index of face: %d    x: %d    y: %d\n" % (second // 60, second % 60, idx, x, y))

    def process(self):
        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                break

            image.flags.writeable = False
            results = self._faceDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            if results.detections:
                for idx, detection in enumerate(results.detections):
                    second = int((int(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000) % 60)

                    relative_bounding_box = detection.location_data.relative_bounding_box
                    image_height, image_width, _ = image.shape
                    absolute_bounding_box = int(relative_bounding_box.xmin * image_width), int(
                        relative_bounding_box.ymin * image_height), int(relative_bounding_box.width * image_width), int(
                        relative_bounding_box.height * image_height)

                    if (second, idx) not in self._detected_moments.keys():
                        self._detected_moments[second, idx] = absolute_bounding_box

                    cv2.rectangle(image, absolute_bounding_box, (255, 0, 255), 2)

            self._result.write(image)
            if cv2.waitKey(10) == ord('q'):
                break

        self._cap.release()
        self._result.release()
        cv2.destroyAllWindows()
        self.write_result_to_file()


def main():
    parser = ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    faceDetector = MyFaceDetector(args.path)
    faceDetector.process()


if __name__ == "__main__":
    main()
