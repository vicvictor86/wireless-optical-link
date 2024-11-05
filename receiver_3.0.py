import cv2
import numpy as np


def recognize_bits(bit_region, threshold=127):
    gray_region = cv2.cvtColor(bit_region, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_region)
    return 1 if mean_intensity > threshold else 0


def is_color_similar(region, color="green"):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    if color == "green":
        lower_bound = np.array([40, 50, 50])
        upper_bound = np.array([80, 255, 255])

        mask = cv2.inRange(hsv_region, lower_bound, upper_bound)

        green_pixels = cv2.countNonZero(mask)
        return green_pixels > (bit_region.size // 4)

    elif color == "red":
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(mask)
        return red_pixels > (region.size // 4)


frames_dir = "frames"

cap = cv2.VideoCapture("https://10.13.75.25:8080/video")

res_original = (1920, 1080)
res_nova = (1280, 720)

scale_w = res_nova[0] / res_original[0]
scale_h = res_nova[1] / res_original[1]

bit_x, bit_y = int(90 * scale_h), int(1000 * scale_w)
timing_x, timing_y = int(90 * scale_h), int(10 * scale_w)
bit_size = int(900 * scale_h)
timing_size = int(900 * scale_w)

string_captured = ""
last_timing_bit = None
can_start_capture = False

frame_count = 0
bit_position = 0
capture_count = 1

green = (0, 255, 0)
red = (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura da webcam.")
        break

    bit_region = frame[bit_x : bit_x + bit_size, bit_y : bit_y + bit_size]
    timing_bit_region = frame[
        timing_x : timing_x + timing_size, timing_y : timing_y + timing_size
    ]

    cv2.rectangle(frame, (bit_y, bit_x), (bit_y + bit_size, bit_x + bit_size), green, 2)
    cv2.rectangle(
        frame,
        (timing_y, timing_x),
        (timing_y + timing_size, timing_x + timing_size),
        red,
        2,
    )

    is_color_similar_to_green = is_color_similar(bit_region, "green")
    is_color_similar_to_red = is_color_similar(bit_region, "red")
    if is_color_similar_to_green:
        can_start_capture = True
        bit_value = "Green Signal Detected"
    elif is_color_similar_to_red:
        if can_start_capture:
            print(f"String Captured: {string_captured}")
            string_captured = ""
            frame_count = 0
            bit_position = 0
            capture_count += 1
        can_start_capture = False
        bit_value = "Red Signal Detected"

    timing_bit = None
    if (
        can_start_capture
        and not is_color_similar_to_green
        and not is_color_similar_to_red
    ):
        timing_bit = recognize_bits(timing_bit_region)
        if last_timing_bit != timing_bit:
            print(f"Timing Bit: {timing_bit} and last Timing Bit: {last_timing_bit}")

            bit_value = recognize_bits(bit_region)
            string_captured += str(bit_value)

            last_timing_bit = timing_bit

    cv2.imwrite(
        f"{frames_dir}/frame_{capture_count}_{bit_position}_{frame_count}.png",
        frame,
    )
    frame_count += 1

    if not can_start_capture:
        bit_value = "Waiting for Green Signal"

    cv2.putText(
        frame,
        f"Bit Value: {bit_value}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        green,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Timing Bit: {timing_bit if can_start_capture and timing_bit is not None else 'N/A'}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        red,
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Reconhecimento de Bits", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("j"):
        bit_y -= 5
    elif key == ord("l"):
        bit_y += 5
    elif key == ord("i"):
        bit_x -= 5
    elif key == ord("k"):
        bit_x += 5
    elif key == ord("a"):
        timing_y -= 5
    elif key == ord("d"):
        timing_y += 5
    elif key == ord("w"):
        timing_x -= 5
    elif key == ord("s"):
        timing_x += 5
    elif key == ord("u"):
        bit_size += 5
    elif key == ord("o"):
        bit_size = max(5, bit_size - 5)
    elif key == ord("y"):
        timing_size += 5
    elif key == ord("h"):
        timing_size = max(5, timing_size - 5)

print(
    f"Coordenadas e Tamanho do Bit Region: Posição ({bit_x}, {bit_y}), Tamanho {bit_size}"
)
print(
    f"Coordenadas e Tamanho do Timing Bit Region: Posição ({timing_x}, {timing_y}), Tamanho {timing_size}"
)

cap.release()
cv2.destroyAllWindows()
