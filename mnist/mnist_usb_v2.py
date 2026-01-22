# -*- coding: utf-8 -*-


from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import xir
import threading
import time
import argparse
from queue import Queue, Empty

os.environ["DISPLAY"] = ":0"

_divider = '-------------------------------'

# ------------------- DPU preprocessing -------------------
def preprocess_fn_img(img, fix_scale):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))
    resized = cv2.resize(gray, (new_w, new_h))

    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    canvas = np.zeros((28,28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = binary

    x = canvas.reshape(28,28,1).astype(np.float32)
    x = x * (1/255.0) * fix_scale
    x = x.astype(np.int8)
    return x

def preprocess_fn_file(image_path, fix_scale):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("? [ERROR] Failed to load image:", image_path)
        return None
    image = cv2.resize(image, (28,28))
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(image == 255) / image.size
    if white_ratio > 0.5:
        image = cv2.bitwise_not(image)
    image = image.reshape(28,28,1).astype(np.float32)
    image = image * (1/255.0) * fix_scale
    image = image.astype(np.int8)
    return image

# ------------------- DPU runner utils -------------------
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    root_subgraph = graph.get_root_subgraph()
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runDPU(id, start, dpu, img):
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids=[]
    ids_max = 10
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    global out_q
    while count < n_of_images:
        runSize = batchSize if count+batchSize<=n_of_images else n_of_images-count
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        for j in range(runSize):
            inputData[0][j,...] = img[(count+j)%n_of_images].reshape(input_ndim[1:])
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start+count))
        count += runSize
        if count<n_of_images and len(ids)<ids_max-1:
            continue
        for index in range(len(ids)):
            dpu.wait((ids[index][0],))  # ? 修正 wait 呼叫
            write_index = ids[index][2]
            for j in range(ids[index][1]):
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids=[]

# ------------------- Image processing -------------------
def process_images(image_dir, threads, model, output_dir="results", dpu_input_dir="IMAGE_DPU_INPUT"):
    VALID_EXT = ('.png','.jpg','.jpeg')
    listimage = [f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXT)]
    runTotal = len(listimage)
    if runTotal == 0:
        print("? [ERROR] No valid images found")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dpu_input_dir, exist_ok=True)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = [vart.Runner.create_runner(subgraphs[0], "run") for _ in range(threads)]
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    img_list = []
    orig_list = []

    for f in listimage:
        path = os.path.join(image_dir, f)
        orig = cv2.imread(path)
        if orig is None:
            continue
        orig_list.append(orig)
        dpu_input = preprocess_fn_file(path, input_scale)
        preview = dpu_input.squeeze().astype(np.float32)
        preview = preview / input_scale * 255.0
        preview = np.clip(preview, 0, 255).astype(np.uint8)
        dpu_img_name = os.path.splitext(f)[0] + "_DPU.jpg"
        cv2.imwrite(os.path.join(dpu_input_dir, dpu_img_name), preview)
        img_list.append(dpu_input)

    # Multi-thread DPU run
    threadAll = []
    start = 0
    for i in range(threads):
        end = len(img_list) if i == threads - 1 else start + len(img_list) // threads
        t = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], img_list[start:end]))
        threadAll.append(t)
        start = end
    t0 = time.time()
    for t in threadAll: t.start()
    for t in threadAll: t.join()
    t1 = time.time()
    fps = len(img_list) / (t1 - t0)
    print(_divider)
    print(f"Throughput={fps:.2f} fps, total frames={len(img_list)}, time={t1-t0:.4f} s")

    # Postprocess & save
    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']
    correct = 0
    wrong = 0
    for i, orig in enumerate(orig_list):
        pred = classes[out_q[i]]
        gt, _ = listimage[i].split('_', 1)
        if gt == pred:
            correct += 1
        else:
            wrong += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, orig.shape[0] - 10)
        cv2.putText(orig, pred, org, font, max(0.6, orig.shape[1]/300.0),
                    (0, 0, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(output_dir, f"{os.path.splitext(listimage[i])[0]}_result.jpg")
        cv2.imwrite(out_path, orig)
    print(f"Correct:{correct}, Wrong:{wrong}, Accuracy:{correct/len(out_q):.4f}")
    print(_divider)

# ------------------- Video processing -------------------
def process_video(video_path, model, output_path="video_result.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("? Cannot open video:", video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    runner = vart.Runner.create_runner(subgraphs[0], "run")
    input_fixpos = runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    global out_q
    out_q = []

    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        box_size = min(h, w) // 2
        x1 = w//2 - box_size//2
        y1 = h//2 - box_size//2
        x2 = x1 + box_size
        y2 = y1 + box_size
        roi = frame[y1:y2, x1:x2]

        dpu_in = [preprocess_fn_img(roi, input_scale)]
        out_q = [None]
        runDPU(0, 0, runner, dpu_in)

        result = out_q[0]
        pred = classes[result] if result is not None else "?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, pred, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        roi_preview = dpu_in[0].astype(np.float32) / input_scale * 255
        roi_preview = roi_preview.reshape(28,28).astype(np.uint8)
        roi_preview_color = cv2.cvtColor(roi_preview, cv2.COLOR_GRAY2BGR)
        overlay_size = 100
        roi_preview_color = cv2.resize(roi_preview_color, (overlay_size, overlay_size))
        frame[h-overlay_size:h, 0:overlay_size] = roi_preview_color

        out.write(frame)

    cap.release()
    out.release()
    print("? Video processing done. Saved to", output_path)
    
class CameraStream:
    def __init__(self, pipeline):
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.q = Queue(maxsize=3)
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.full(): self.q.put(frame)
    def read(self): return self.q.get() if not self.q.empty() else None
    def stop(self): self.stopped = True; self.cap.release()

# ------------------- Live Camera processing -------------------
def process_camera(model):
    # 環境變數與顯示設定

    WIDTH_IN, HEIGHT_IN, FPS = 1280, 720, 30
    #WIDTH_OUT, HEIGHT_OUT = 900, 675
    WIDTH_OUT, HEIGHT_OUT = 1280, 720
    cam_type = 'usb'

    # 注意：請確保 WIDTH_IN, HEIGHT_IN, FPS 等變數已在上方定義
    stream = CameraStream(f"v4l2src device=/dev/video0 ! image/jpeg, width={WIDTH_IN}, height={HEIGHT_IN}, framerate={FPS}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false").start()
    out = cv2.VideoWriter(f"appsrc is-live=true format=TIME ! video/x-raw, format=BGR, width={WIDTH_OUT}, height={HEIGHT_OUT}, framerate={FPS}/1 ! videoconvert ! ximagesink sync=false", cv2.CAP_GSTREAMER, 0, float(FPS), (WIDTH_OUT, HEIGHT_OUT), True)

    

    # ===== Load DPU model =====
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_fixpos = runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    classes = ['zero','one','two','three','four',
               'five','six','seven','eight','nine']

    global out_q
    out_q = [None]

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            frame = stream.read()
            if frame is None: continue
            
            h, w, _ = frame.shape
            box_size = min(h, w) // 4  # 框框大小
            x1 = w//2 - box_size//2
            y1 = h//2 - box_size//2
            x2 = x1 + box_size
            y2 = y1 + box_size
            roi = frame[y1:y2, x1:x2]

            # ===== DPU 預處理 + 推論 =====
            dpu_in = [preprocess_fn_img(roi, input_scale)]
            out_q = [None]

            input_ndim = tuple(runner.get_input_tensors()[0].dims)
            output_ndim = tuple(runner.get_output_tensors()[0].dims)
            inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
            inputData[0][0,...] = dpu_in[0].reshape(input_ndim[1:])
            outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]

            job_id = runner.execute_async(inputData, outputData)
            runner.wait(job_id)

            out_q[0] = np.argmax(outputData[0][0])
            pred = classes[out_q[0]] if out_q[0] is not None else "?"

            # ===== 畫結果 =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, pred, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # ===== DPU input preview =====
            roi_preview = dpu_in[0].astype(np.float32) / input_scale * 255
            roi_preview = roi_preview.reshape(28,28).astype(np.uint8)
            roi_preview_color = cv2.cvtColor(roi_preview, cv2.COLOR_GRAY2BGR)
            overlay_size = 100
            roi_preview_color = cv2.resize(roi_preview_color, (overlay_size, overlay_size))
            frame[h-overlay_size:h, 0:overlay_size] = roi_preview_color

#             # ===== 計算 FPS =====
#             frame_count += 1
#             if frame_count % 30 == 0:
#                 now = time.time()
#                 fps = 30 / (now - last_time)
#                 last_time = now
#                 print(f"?? Current FPS: {fps:.2f}")

#             # 預設 fps
#             if 'fps' not in locals():
#                 fps = 0.0

#             # ===== 在左上角顯示 FPS =====
#             cv2.putText(frame, f"FPS: {fps:.2f}",
#                         (10, 30),  # 左上角
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (255,0,0), 1)

            # ===== 輸出到 HDMI =====
            out.write(frame)

            # q 離開
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("?? USER STOP (Ctrl+C)")

    finally:
        out.release()
        cv2.destroyAllWindows()
        print("? Camera HDMI END PROGRAM")

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d','--image_dir',type=str,default='images')
    ap.add_argument('-t','--threads',type=int,default=1)
    ap.add_argument('-m','--model',type=str,default='CNN_kv260_4096.xmodel')
    ap.add_argument('-v','--video',type=str,help='Input video path')
    
    # ?? 修改點：將 action='store_true' 替換為 type=str 和 nargs='?'
    # nargs='?': 允許參數是可選的。如果提供了 -c 但沒有值，它會設置為 'mipi' (default)。
    # const='mipi': 當只提供 -c/--camera 而沒有值時，給予的默認值。
    ap.add_argument('-c', '--camera', action = 'store_true', 
                    help='Use live camera. Specify type: "usb" .')
    
    
    args = ap.parse_args()

    # ... 打印輸出 (可選) ...
    print("Command line options:")
    print(" --image_dir :", args.image_dir)
    print(" --threads   :", args.threads)
    print(" --model     :", args.model)
    if args.video:
        print(" --video     :", args.video)
    if args.camera:
        # ?? 輸出 args.camera 的值 (即 'usb' 或 'mipi')
        print(" --camera    :", args.camera) 

    if args.camera:
        # ?? 根據 camera_type 呼叫不同的處理函數或傳遞參數
        process_camera(args.model) 
    elif args.video:
        process_video(args.video, args.model)
    else:
        process_images(args.image_dir, args.threads, args.model)

if __name__=="__main__":
    main()
