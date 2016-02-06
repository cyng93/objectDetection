#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from common import clock, draw_str


help_message = '''
USAGE: objdetect.py [--cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import os, sys, getopt
    print help_message

    args, detect_src_dir = getopt.getopt(sys.argv[1:], '', ['cascade='])
    try: detect_src_dir = detect_src_dir[0]
    except: detect_src_dir = 0
    args = dict(args)
    cascade_fn = args.get('--cascade')

    cascade = cv2.CascadeClassifier(cascade_fn)

    for root, subdirs, files in os.walk(detect_src_dir):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(root, filename)
                print('\t- file %s (full path: %s)' % (filename, file_path))

                detect_src = file_path
                img = cv2.imread(detect_src)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                t = clock()
                rects = detect(gray, cascade)
                vis = img.copy()
                draw_rects(vis, rects, (0, 255, 0))
                dt = clock() - t

                draw_str(vis, (20, 20), 'classifier: %s' % cascade_fn)
                draw_str(vis, (20, 40), 'time: %.1f ms' % (dt*1000))
                #cv2.imshow('objdetect', vis)
                cv2.imwrite('_result/'+filename, vis)

                #while True:
                #    if 0xFF & cv2.waitKey(5) == 27:
                #        break

            else:
                continue

    #cv2.destroyAllWindows()
