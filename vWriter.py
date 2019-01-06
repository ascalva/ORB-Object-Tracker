import cv2

"""
Authors: Stephen Kim, Alberto Serrano
"""
class VideoWriterWrapper:
    def __init__(self, frame_width, frame_height, fps=30.0, fn="output.avi"):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter(fn,fourcc,fps,(frame_width,frame_height))

    def write(self, frame):
        self.out.write(frame)

    def cleanup(self):
        self.out.release()
