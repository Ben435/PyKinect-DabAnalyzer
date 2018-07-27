import freenect
import cv2
import numpy as np
from os import path
from threadsafe_vars import AtomicTimeVar
from avg_recorder import AvgRecorder

initialized = False

cur_cam_angle = 15
max_angle = 30
min_angle = 0

subtract_background = False
max_contours = 1
min_contour_area = 10000
dist_map_offset = [-20, 5]


def set_cam_angle(dev, new_angle):
    if min_angle <= new_angle <= max_angle:
        freenect.set_tilt_degs(dev, new_angle)
        print("New Angle: {}".format(new_angle))
        return new_angle
    else:
        if min_angle > new_angle:
            print("New angle {} is lower than min angle {}".format(new_angle, min_angle))
            return min_angle
        elif max_angle < new_angle:
            print("New angle {} is higher than max angle {}".format(new_angle, max_angle))
            return max_angle
        else:
            raise ValueError("Wat? failed: {} <= {} <= {}".format(min_angle, new_angle, max_angle))


def gen_handle_depth(atomic_depth_var):
    def handle_depth(dev, depth, timestamp):
        array = depth.astype(np.uint8)

        # Set
        atomic_depth_var.set_val(array, timestamp)

    return handle_depth


def gen_handle_video(atomic_color_var, atomic_gray_var):
    def handle_video(dev, video, timestamp):
        bw = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
        color = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

        atomic_color_var.set_val(color, timestamp)
        atomic_gray_var.set_val(bw, timestamp)

    return handle_video


def gen_handle_body(face_cascade, atomic_color_var, atomic_gray_var, atomic_depth_var):
    avg_recorder = AvgRecorder(150)

    def handle_body(dev, ctx):
        global cur_cam_angle, initialized, subtract_background

        # Init settings.
        if not initialized:
            print("Initializing...")
            set_cam_angle(dev, cur_cam_angle)
            print("Done!")
            initialized = True

        raw_depth = atomic_depth_var.get_val()
        depth_frame = None
        if raw_depth is not None:
            if not avg_recorder.initialized:
                avg_recorder.initialize(raw_depth.shape)
            if avg_recorder.is_recording():
                avg_recorder.record(raw_depth)

            depth_frame = np.copy(raw_depth)
            if subtract_background:
                avg_map = avg_recorder.get_avg()
                if avg_map is not None:
                    # Subtract avg background...
                    subtracted_frame = np.greater(depth_frame, avg_map)
                    depth_frame = np.where(subtracted_frame,
                                            np.zeros(depth_frame.shape),
                                            depth_frame).astype(np.uint8)
                # if background_min_depth is not None:
                #     # Threshold background
                #     #print(background_min_depth)
                #     cv2.threshold(depth_frame, background_min_depth, 255, cv2.THRESH_BINARY, dst=depth_frame)

            # display depth image
            cv2.imshow('Depth image', depth_frame)

        frame = atomic_color_var.get_val()
        grey_frame = atomic_gray_var.get_val()
        if frame is not None and grey_frame is not None:

            # Draw contours on rgb
            ret, thresh = cv2.threshold(depth_frame, 1, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out tiny contours.
            cnt_area = filter(lambda a: a[1] > min_contour_area,
                              sorted(
                                  map(lambda cnt: tuple([cnt, cv2.contourArea(cnt)]), contours),
                                  key=lambda a: a[1],
                                  reverse=True))[:max_contours]

            for cnt, area in cnt_area:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(frame, 'Cnt({})'.format(area), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

                # Detect objects
                c_x, c_y, c_w, c_h = cv2.boundingRect(cnt)
                reduced_greyscale = grey_frame[c_y:c_y+c_h, c_x:c_x+c_w]

                # Faces...
                faces = face_cascade.detectMultiScale(reduced_greyscale, 1.3, 5)
                for (x, y, w, h) in faces:
                    # Draw boxes with offset.
                    frame = cv2.rectangle(frame, (c_x+x, c_y+y), (c_x+x + w, c_y+y + h), (255, 0, 0), 2)

                # Transform and draw.
                cv2.drawContours(frame,
                                 np.add(cnt, dist_map_offset),
                                 -1,
                                 (0, 255, 0))
            # display RGB image
            cv2.imshow('RGB image', frame)

        # Wait 5 ms for keys.
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            if k == 27:
                # quit program when 'esc' key is pressed
                cv2.destroyAllWindows()
                raise freenect.Kill
            elif k == 43:
                # plus = sensor up
                cur_cam_angle = set_cam_angle(dev, cur_cam_angle + 1)
            elif k == 45:
                # minus = sensor down
                cur_cam_angle = set_cam_angle(dev, cur_cam_angle - 1)
            elif k == 80:
                # home = begin background record
                avg_recorder.begin_record()
            elif k == 102:
                # f = toggle background subtraction
                subtract_background = not subtract_background
            else:
                print(k)

    return handle_body


def main():
    face_cascade = cv2.CascadeClassifier(path.join(cv2.haarcascades, "haarcascade_frontalface_default.xml"))

    color_img = AtomicTimeVar()
    gray_img = AtomicTimeVar()
    depth_img = AtomicTimeVar()

    freenect.runloop(depth=gen_handle_depth(depth_img),
                     video=gen_handle_video(color_img, gray_img),
                     body=gen_handle_body(face_cascade, color_img, gray_img, depth_img))


if __name__ == "__main__":
    main()
