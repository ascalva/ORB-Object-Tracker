"""
Implements an ORB-based object tracker as specified by the paper:
   Object Tracking Based on ORB and Temporal-Spacial Constraint by Shuang Wu,
   IEEE Student Member, Yawen Fan, Shibao Zheng, IEEE Member and Hua Yang, IEEE
   Member

Authors: Alberto Serrano, Stephen Kim

"""

import time
import cv2
import random
# from vWriter import VideoWriterWrapper
import numpy as np

# Define global variables
M = (0,0)
centers = []

# def liveFeedMatches():
#     global vidWriter
#     cap = cv2.VideoCapture(0)#"IMG_3385.MOV")
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     vidWriter = VideoWriterWrapper(frame_width, frame_height)
#     time.sleep(0.3)
#     prevImg = None
#
#     # always comparing two frames, so do something else seeing first frame
#     # prompt for ROI, then set to cframe
#     ret, img = cap.read()
#     # TODO: return to selectROI
#     bbox = cv2.selectROI(img, False)
#     #bbox = (603, 322, 136, 214) # hard coded ROI for IMG_3385.MOV
#     prevImg = img
#     x = int(bbox[0])
#     y = int(bbox[1])
#     w = int(bbox[2])
#     h = int(bbox[3])
#     x_i = x + w/2
#     y_i = y + h/2
#     cframe = (x_i, y_i, w, h)
#     pframe = cframe
#     while(True):
#         # Capture frame-by-frame
#         ret, img = cap.read()
#
#         # constraint to end when reading from a video file instead of a device
#         # video stream
#         if img is None:
#             return
#
#         else:
#             pframe, cframe = processLiveFeed(prevImg, img, pframe, cframe)
#             prevImg = img
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # When everything done, release the capture
#     cap.release()
#     vidWriter.cleanup()
#     # cv2.destryoAllWindows()

"""
given two frames, previous and current frame, determine the search frame.

This function directly reflects formulas (9), (10), and (11) in the paper. What
gets returned in a 4-tuple representing S_i+1 from the paper

:param: prev (4-tuple) specifying a frame conforming to the following:
   1. x coordinate of the center of the frame
   2. y coordinate of the center of the frame
   3. width of the frame
   4. height of the frame
:param: curr (4-tuple) specifying a frame conforming to the following:
   1. x coordinate of the center of the frame
   2. y coordinate of the center of the frame
   3. width of the frame
   4. height of the frame
:param: ap (int) specifying an alpha constant; used to increase the search
    frame's width and height
:return: (4-tuple) specifying the new search frame
"""
def getSearchFrame(prev, curr, ap = 5):
    w = curr[2]
    h = curr[3]
    w_ = w + ap
    h_ = h + ap
    m_x = curr[0] - prev[0]
    m_y = curr[1] - prev[1]
    u_i = curr[0] + m_x
    v_i = curr[1] + m_y
    return(u_i, v_i, w_, h_)

# [c_x, c_y, w, h] -> [x, y, w, h]
"""
given a frame, return a bounding box

:param: frame (4-tuple) specifying a frame conforming to the following:
   1. x coordinate of the center of the frame
   2. y coordinate of the center of the frame
   3. width of the frame
   4. height of the frame

:return: (4-tuple) specifying a bounding box conforming to the following:
   1. x coordinate for top left of the bounding box
   2. y coordinate for top left of the bounding box
   3. width of the frame
   4. height of the frame
"""
def bboxFromFrame(frame):
    x = frame[0]
    y = frame[1]
    w = frame[2]
    h = frame[3]
    return (int(x - (w/2)), int(y - (h/2)), w, h)

"""
Processes an image with two frames of the same object at different times to
determine the most probable location of the next frame for the object. Function
returns two 4-tuples describing the previous frame (which is just the current
frame) and the next frame

:param: cur (nd np.array, an image) image corresponding to cframe
:param: nxt (nd np.array, an image) image corresponding to s_i, the search frame
:param: pframe (4-tuple) specifying the current frame conforming to the
following
   1. x coordinate of the center of the frame
   2. y coordinate of the center of the frame
   3. width of the frame
   4. height of the frame
:param: cframe (4-tuple) specifying the current frame conforming to the
following
   1. x coordinate of the center of the frame
   2. y coordinate of the center of the frame
   3. width of the frame
   4. height of the frame
:return: two 4-tuples; specifying the next previous and current frames
"""
def processLiveFeed(cur, nxt, pframe, cframe):
    global M
    orb = cv2.ORB_create(1000, 1.2)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    s_i = getSearchFrame(pframe, cframe)

    framebbox = bboxFromFrame(cframe)
    sbbox = bboxFromFrame(s_i)


    x = framebbox[0]
    y = framebbox[1]
    w = framebbox[2]
    h = framebbox[3]
    kp1,  des1 = orb.detectAndCompute(
        cur[y:y+h, x:x+w], None
    )

    s_x = sbbox[0]
    s_y = sbbox[1]
    s_w = sbbox[2]
    s_h = sbbox[3]
    kp2, des2 = orb.detectAndCompute(
        nxt[s_y:s_y+s_h, s_x:s_x+s_w], None
    )
    a = (x,y)
    b = (s_x, s_y)
    frame_i = cur[y:y+h, x:x+w]
    frame_ipp = nxt[s_y:s_y+s_h, s_x:s_x+s_w]

    if not ((des1 is None) or (des2 is None)):
        matches   = bf.match(des1,des2)
        matches   = sorted(matches, key=lambda val: val.distance)

        M = videoDrawMatches(frame_i, a, kp1, frame_ipp, b, kp2, matches, 0, framebbox, cur, s_i)
    # kp1, des1 = kp2, des2
    # print(M)
    nframe = (cframe[0] + int(M[0]), cframe[1] + int(M[1]), cframe[2], cframe[3])
    return cframe, nframe

"""
Probably gonna change in future commits but heres some documentation:

img1 - cropped image of "cur"
img1_coord - bounding box of "cur"
kp1 - keypoints of img1
img2 - cropped image of "nxt"
img2_coord - bounding box of "nxt"
kp2 - keypoints of img2
matches - matches between img1 and img2
"""
def videoDrawMatches(img1, img1_coord, kp1, img2, img2_coord, kp2, matches, counter, bbox, out, s_i, n_key = 3):
    global vidWriter
    #out = img1

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    x = int(bbox[0])
    y = int(bbox[1])
    C1_x = 0
    C1_y = 0
    C2_x = 0
    C2_y = 0
    for mat in matches[:n_key]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1)  = get_real_coordinate(kp1[img1_idx].pt, img1_coord)
        (x2,y2)  = get_real_coordinate(kp2[img2_idx].pt, img2_coord)
        C1_x += x1
        C1_y += y1
        C2_x += x2
        C2_y += y2

        # Draw circles around keypoints
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 1)

    C1_x /= n_key
    C1_y /= n_key
    C2_x /= n_key
    C2_y /= n_key

    # TODO: debugging purposes, dispose of when no longer needed
    print("Train: (" + str(C1_x) + ",", str(C1_y)+ ")")
    print("Query: (" + str(C2_x) + ",", str(C2_y)+ ")")
    print("Motion: (" + str(C2_x-C1_x) + ",", str(C2_y-C1_y)+ ")")
    print()

    centers.append((s_i[0],s_i[1]))
    for i in range(len(centers)):
        cv2.circle(out, (round(centers[i][0]), round(centers[i][1])),
            1, (0, 255, 0), 4
        )

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(out, p1, p2, (255,0,0), 2, 1)

    # if you want to see ORB keypoints and matches at each iteration
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:n_key],None,flags=2)
    #cv2.imshow("Top " + str(n_key) + " ORB Keypoints and matched", img3)

    # if you want to see the whole scene and watch the bounding box move
    cv2.imshow("Object tracking", out)
    if vidWriter is not None:
        vidWriter.write(out)

    # cv2.imwrite("track_path.png", out)

    return C2_x - C1_x, C2_y - C1_y

"""
Takes a coordinate from within a cropped image (e.g. a keypoint coordinate), a
frame specifying where the cropped image exists in relation to the scene, and
returns the "true" coordinate as it would exist in the scene.

:param: coord (2-tuple) - specifies (x,y) coordinate from within a cropped image
:param: frame (4-tuple) - specifies (x,y,w,h) bounding box that represents the
    cropped image in a scene
:return: (2-tuple) the x,y coordinate in terms of the scene

This function proves useful in calculating the motion vector. Since ORB
keypoints and descriptors are determined based on a cropped version of a larger
scene (for performance reasons) when calculating the motion vector between two
images, the keypoints need to be converted back to the real coordinates in the
scene, otherwise the motion vectors are calculating relative distances which
don't accurately reflect the change in centers
"""
def get_real_coordinate(coord,frame):
    return frame[0]+coord[0], frame[1]+coord[1]

def main():
    liveFeedMatches()
