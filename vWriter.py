import cv2

class VideoWriterWrapper:
    def __init__(self, frame_width, frame_height, fps=30.0, fn="output.avi"):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter(fn,fourcc,fps,(frame_width,frame_height))

    def write(self, frame):
        self.out.write(frame)

    def cleanup(self):
        self.out.release()

def main():
    print()
    cap = cv2.VideoCapture('IMG_3385.MOV')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        #print(frame.shape)
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
