import cv2
import numpy as np
import time


def show_color(color, display_time=2):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:, :] = color

    cv2.imshow("Transmitter", img)
    time.sleep(display_time)

    cv2.waitKey(1)


def transition_colors(sequence, previous_bit_timing_color_was_black, bit_timing=2):
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    show_color(green, bit_timing)

    for char in sequence:
        if char == "1":
            color = white
        elif char == "0":
            color = black

        img = np.zeros((500, 500, 3), dtype=np.uint8)

        img[:, 200:] = color

        previous_bit_timing_color_was_black = not previous_bit_timing_color_was_black
        img[:, :200] = white if previous_bit_timing_color_was_black else black

        cv2.imshow("Transmitter", img)

        time.sleep(bit_timing)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    show_color(red, bit_timing)

    return True, previous_bit_timing_color_was_black


def centralize_window(window_name, width, height):
    screen_width = 1920
    screen_height = 1080

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    cv2.moveWindow(window_name, x, y)


def run_transmitter(sequence):
    width = 1200
    height = 1080

    cv2.namedWindow("Transmitter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Transmitter", width, height)
    centralize_window("Transmitter", width, height)

    continue_program = True
    previous_bit_timing_color_was_black = False
    while continue_program:
        continue_program, previous_bit_timing_color_was_black = transition_colors(
            sequence,
            previous_bit_timing_color_was_black,
            bit_timing=2,
        )

    cv2.destroyAllWindows()


transmitter_message = "010"
run_transmitter(transmitter_message)
