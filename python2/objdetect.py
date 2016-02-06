#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from common import clock, draw_str


help_message = '''
USAGE: objdetect.py [--cascade <cascade_fn>] [--input <input_dir>] [--output <output_dir>]
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

    args, detect_src_dir = getopt.getopt(sys.argv[1:], '', ['cascade=', 'input=', 'output='])
    #try: detect_src_dir = detect_src_dir[0]
    #except: detect_src_dir = 0
    args = dict(args)
    cascade_fn = args.get('--cascade')
    input_dir = args.get('--input')
    output_dir = args.get('--output')

    cascade = cv2.CascadeClassifier(cascade_fn)

    try: os.stat(output_dir)
    except: os.mkdir(output_dir)

    for root, subdirs, files in os.walk(input_dir):
        for subdir in subdirs:
            try:
                os.stat(os.path.join(output_dir, subdir))
            except:
                os.mkdir(os.path.join(output_dir, subdir))

        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(root, filename)
                print('\t- file %s (full path: %s)' % (filename, file_path))

                subdir = file_path.split('/')[1]

                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                t = clock()
                rects = detect(gray, cascade)
                vis = img.copy()
                draw_rects(vis, rects, (0, 255, 0))
                dt = clock() - t

                draw_str(vis, (10, 20), 'classifier: %s' % cascade_fn)
                draw_str(vis, (10, 40), 'time: %.1f ms' % (dt*1000))
                #cv2.imshow('objdetect', vis)
                cv2.imwrite(os.path.join(output_dir, subdir, filename), vis)

                #while True:
                #    if 0xFF & cv2.waitKey(5) == 27:
                #        break

            else:
                continue

    #cv2.destroyAllWindows()
