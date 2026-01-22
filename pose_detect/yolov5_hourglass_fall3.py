# -*- coding: utf-8 -*-
import numpy as np
import cv2
import vart
import xir
import os
import time
import threading
from queue import Queue, Empty
from collections import deque

# 環境變數與顯示設定
os.environ["DISPLAY"] = ":0"
WIDTH_IN, HEIGHT_IN, FPS = 1280, 720, 30
#WIDTH_OUT, HEIGHT_OUT = 900, 675
WIDTH_OUT, HEIGHT_OUT = 1280, 720

def sigmoid(x):
    return 1 / (1 + np.exp(x * -1))

SKELETON_LINES_16 = [
    (9, 8), (8, 7), (7, 6), (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15), (6, 2), (2, 1), (1, 0),
    (6, 3), (3, 4), (4, 5)
]

# --- 全域線程通訊變數 ---
pose_task_queue = Queue(maxsize=20) # 存放 Hourglass 任務
pose_results = {}                    # 存放最新骨架結果: {pid: {'kpts': [], 'status': "", 'box': []}}
results_lock = threading.Lock()      # 確保讀寫字典安全

# --- 模型類別 (保持你提供的邏輯) ---

class YOLOv5Detector:
    def __init__(self, model_path, conf_thresh=0.35, iou_thresh=0.45):
        self.graph = xir.Graph.deserialize(model_path)
        self.runner = vart.Runner.create_runner([s for s in self.graph.get_root_subgraph().toposort_child_subgraph() if s.has_attr("device") and s.get_attr("device").upper() == "DPU"][0], "run")
        self.in_scale = 2**self.runner.get_input_tensors()[0].get_attr("fix_point")
        self.out_scales = [2**t.get_attr("fix_point") for t in self.runner.get_output_tensors()]
        self.input_h, self.input_w = self.runner.get_input_tensors()[0].dims[1], self.runner.get_input_tensors()[0].dims[2]
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.anchors = np.array([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]])
        self.strides = np.array([8, 16, 32])

    def preprocess(self, img):
        h, w, _ = img.shape
        scale = min(self.input_h / h, self.input_w / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        dx, dy = (self.input_w - nw) // 2, (self.input_h - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw, :] = resized
        return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) * (self.in_scale / 255.0)).astype(np.int8), scale, dx, dy

    def postprocess(self, dpu_outputs, scale, pad_w, pad_h):
        all_boxes = []
        conf_logit_thresh = -np.log(1/(self.conf_thresh - 1e-6) - 1)
        for i, output in enumerate(dpu_outputs):
            data = output[0]
            if len(data.shape) == 3: data = data.reshape(data.shape[0], data.shape[1], 3, -1)
            mask = data[..., 4] > (conf_logit_thresh * self.out_scales[i])
            if not np.any(mask): continue
            valid_data = data[mask].astype(np.float32) / self.out_scales[i]
            grid = np.argwhere(mask)
            y, x, a = grid[:, 0], grid[:, 1], grid[:, 2]
            xy = (sigmoid(valid_data[:, 0:2]) * 2.0 - 0.5 + np.stack([x, y], axis=-1)) * self.strides[i]
            wh = (sigmoid(valid_data[:, 2:4]) * 2.0)**2 * self.anchors[i][a]
            scores = sigmoid(valid_data[:, 4]) * sigmoid(valid_data[:, 5])
            s_mask = scores > self.conf_thresh
            if not np.any(s_mask): continue
            boxes = np.zeros((np.sum(s_mask), 5))
            boxes[:, 0] = (xy[s_mask, 0] - wh[s_mask, 0]/2 - pad_w) / scale
            boxes[:, 1] = (xy[s_mask, 1] - wh[s_mask, 1]/2 - pad_h) / scale
            boxes[:, 2] = (xy[s_mask, 0] + wh[s_mask, 0]/2 - pad_w) / scale
            boxes[:, 3] = (xy[s_mask, 1] + wh[s_mask, 1]/2 - pad_h) / scale
            boxes[:, 4] = scores[s_mask]
            all_boxes.append(boxes)
        return self.nms(np.vstack(all_boxes)) if all_boxes else []

    def nms(self, boxes):
        x1, y1, x2, y2, scores = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]; keep = []
        while order.size > 0:
            i = order[0]; keep.append(boxes[i])
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= self.iou_thresh)[0] + 1]
        return keep

class HourglassDetector:
    def __init__(self, model_path):
        self.graph = xir.Graph.deserialize(model_path)
        self.runner = vart.Runner.create_runner([s for s in self.graph.get_root_subgraph().toposort_child_subgraph() if s.has_attr("device") and s.get_attr("device").upper() == "DPU"][0], "run")
        self.in_scale = 2**self.runner.get_input_tensors()[0].get_attr("fix_point")
        self.in_h, self.in_w = self.runner.get_input_tensors()[0].dims[1:3]

    def run(self, roi):
        roi_h, roi_w = roi.shape[:2]
        resized = cv2.resize(roi, (self.in_w, self.in_h))
        input_data = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) * (self.in_scale / 255.0)).astype(np.int8)
        in_buf = [np.empty((1, self.in_h, self.in_w, 3), dtype=np.int8, order="C")]
        in_buf[0][0, ...] = input_data
        out_buf = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in self.runner.get_output_tensors()]
        self.runner.wait(self.runner.execute_async(in_buf, out_buf))
        heatmaps = out_buf[0][0].astype(np.float32)
        hm_h, hm_w, num_points = heatmaps.shape
        kpts = []
        CONF_THRESH = 20 
        for i in range(num_points):
            single_hm = heatmaps[:, :, i]
            max_val = np.max(single_hm)
            if max_val > CONF_THRESH:
                idx = np.argmax(single_hm)
                y, x = np.unravel_index(idx, (hm_h, hm_w))
                if x < 2 or y < 2 or x > hm_w-3 or y > hm_h-3:
                    kpts.append(None)
                else:
                    kpts.append((int(x * (roi_w / hm_w)), int(y * (roi_h / hm_h))))
            else:
                kpts.append(None)
        return kpts
        
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

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_age_seconds=1.5):
        self.prev_persons = {} # pid: [box, last_time, confidence]
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age_seconds = max_age_seconds

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, current_boxes_with_conf):
        
        now = time.time()
        new_persons = {}
        matched_ids = set()

        for cur_det in current_boxes_with_conf:
            cur_box = cur_det[:4]
            cur_conf = cur_det[4]
            max_iou, best_id = 0, None

            for pid, (prev_box, last_time, prev_conf) in self.prev_persons.items():
                if pid in matched_ids: continue
                iou = self.calculate_iou(cur_box, prev_box)
                if iou > max_iou:
                    max_iou, best_id = iou, pid

            if max_iou > self.iou_threshold:
                new_persons[best_id] = [cur_box, now, cur_conf]
                matched_ids.add(best_id)
            else:
                new_persons[self.next_id] = [cur_box, now, cur_conf]
                self.next_id += 1

        # 保留消失但未超時的 ID
        for pid, data in self.prev_persons.items():
            if pid not in matched_ids:
                if now - data[1] < self.max_age_seconds:
                    new_persons[pid] = data # 沿用舊 box 與 conf

        self.prev_persons = new_persons
        # 返回格式: {pid: [box, confidence]}
        return {pid: [d[0], d[2]] for pid, d in new_persons.items() if d[1] == now}
        
        


class FallDetector:
    def __init__(self):
        self.history = {} 
        self.WINDOW_SIZE = 60
        self.LOOK_BACK_TIME = 0.65
        self.ANGLE_DELTA_THRESH = 70 
        self.DROP_THRESH = 30         
        self.STILL_THRESH = 3.0       
        self.HEIGHT_SHRINK_THRESH = 0.65 
        self.SEVERE_SHRINK_THRESH = 0.65 

    def check(self, pid, kpts, box):
        now = time.time()
        x1, y1, x2, y2 = box
        curr_h = y2 - y1 

        if pid not in self.history:
            self.history[pid] = {
                'data': deque(maxlen=self.WINDOW_SIZE), 
                'state': "Normal", 
                'trigger_time': None, 
                'action_confirmed': False,
                'baseline_h': None,      
                'confirm_elapsed': 0.0,
            }
        
        p7, p6 = kpts[7], kpts[6]
        angle, head_y = None, None
        if p7 and p6:
            angle = abs(np.degrees(np.arctan2(p7[1] - p6[1], p7[0] - p6[0])))
            head_y = p7[1]

        # --- 關鍵修改 1：儲存 y1 與 y2 到歷史紀錄 ---
        # 格式：(時間, 角度, 頭點y, 高度, y1, y2)
        self.history[pid]['data'].append((now, angle, head_y, curr_h, y1, y2))
        data_queue = self.history[pid]['data']
        
        past_data = None
        for i in range(len(data_queue) - 1, -1, -1):
            if now - data_queue[i][0] >= self.LOOK_BACK_TIME:
                past_data = data_queue[i]
                break
        
        if past_data is not None:
            h_shrink_ratio = curr_h / (past_data[3] + 1e-6)
            
            # --- 關鍵修改 2：計算位移向量 ---
            past_y1, past_y2 = past_data[4], past_data[5]
            y1_drop = y1 - past_y1  # 正值：頂邊下移（跌倒特徵）
            y2_up = past_y2 - y2    # 正值：底邊上縮（遮蔽特徵）

            # --- 1. 觸發偵測 ---
            if not self.history[pid]['action_confirmed']:
                triggered = False
                
                # A 條件：有姿態點時 (強化：檢查頭部點是否真的下墜)
                if head_y is not None and past_data[2] is not None:
                    y_diff = head_y - past_data[2]
                    angle_diff = abs(angle - past_data[1]) if (angle and past_data[1]) else 0
                    if y_diff > self.DROP_THRESH and (angle_diff > self.ANGLE_DELTA_THRESH or h_shrink_ratio < self.HEIGHT_SHRINK_THRESH):
                        triggered = True
                
                # B 條件：沒姿態點時 (增加向上/向下縮的邏輯判斷)
                elif h_shrink_ratio < self.SEVERE_SHRINK_THRESH:
                    # 只有當「頂邊下移幅度」大於「底邊上縮幅度」，且頂邊有實質下移時，才判定為跌倒
                    # 這樣可以有效過濾掉人走過桌子後方（底邊上縮）導致的高縮減比例
                    if y1_drop > y2_up and y1_drop > 20:
                        triggered = True

                if triggered:
                    self.history[pid]['action_confirmed'] = True
                    self.history[pid]['trigger_time'] = now
                    self.history[pid]['baseline_h'] = past_data[3]
                    self.history[pid]['state'] = "CONFIRM"

            # --- 2. 判定與復歸邏輯 ---
            else:
                elapsed = now - self.history[pid]['trigger_time']
                self.history[pid]['confirm_elapsed'] = elapsed 
                
                base_h = self.history[pid]['baseline_h']
                recovery_ratio = curr_h / (base_h + 1e-6)
                
                # 如果高度恢復，或頂邊重新回到高位，則復歸
                if recovery_ratio > 0.85:
                    self.history[pid].update({
                        'action_confirmed': False, 'state': "Normal", 
                        'trigger_time': None, 'baseline_h': None, 'confirm_elapsed': 0.0
                    })
                elif elapsed > self.STILL_THRESH:
                    self.history[pid]['state'] = "FALL"
                        
        return self.history[pid]['state'], self.history[pid].get('confirm_elapsed', 0.0)

# --- 背景線程工作邏輯 (配合修改調用方式) ---
def pose_worker(model_path):
    pose_executor = HourglassDetector(model_path)
    
    
    
    while True:
        try:
            # 任務內容現在包含信心值 conf
            task = pose_task_queue.get(timeout=1)
            if task is None: break
            pid, roi, x_s, y_s, box, conf = task # 這裡多接收一個 conf
            
            # 執行姿態辨識
            kpts = pose_executor.run(roi)
            # 轉換為全域座標
            kpts_g = [(p[0] + x_s, p[1] + y_s) if p else None for p in kpts]
            
            # 更新結果字典
            with results_lock:
                pose_results[pid] = {
                    'kpts': kpts_g,  
                    'box': box,
                    'conf': conf  # 將信心值存回結果，供主程式繪製
                }
            pose_task_queue.task_done()
        except Empty:
            continue

# --- 主程式修正版 ---
if __name__ == "__main__":
    yolo = YOLOv5Detector("yolov5_nano_pt.xmodel")
    tracker = IoUTracker(iou_threshold=0.3)
    fall_logic = FallDetector()
    # 設定確認靜止的時間門檻
    fall_logic.STILL_THRESH = 3.0 
    
    # 建立 Hourglass 線程池
    for _ in range(20): 
        threading.Thread(target=pose_worker, args=("hourglass-pe_mpii.xmodel",), daemon=True).start()

    # 注意：請確保 WIDTH_IN, HEIGHT_IN, FPS 等變數已在上方定義
    stream = CameraStream(f"v4l2src device=/dev/video0 ! image/jpeg, width={WIDTH_IN}, height={HEIGHT_IN}, framerate={FPS}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false").start()
    out = cv2.VideoWriter(f"appsrc is-live=true format=TIME ! video/x-raw, format=BGR, width={WIDTH_OUT}, height={HEIGHT_OUT}, framerate={FPS}/1 ! videoconvert ! ximagesink sync=false", cv2.CAP_GSTREAMER, 0, float(FPS), (WIDTH_OUT, HEIGHT_OUT), True)

    frame_count, last_time, fps_display = 0, time.time(), 0

    try:
        while True:
            frame = stream.read()
            if frame is None: continue

            # 1. YOLOv5 推論
            input_data, scale, dx, dy = yolo.preprocess(frame)
            in_buf = [np.empty(tuple(yolo.runner.get_input_tensors()[0].dims), dtype=np.int8, order="C")]
            in_buf[0][0, ...] = input_data
            out_buf = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in yolo.runner.get_output_tensors()]
            yolo.runner.wait(yolo.runner.execute_async(in_buf, out_buf))
            persons = yolo.postprocess(out_buf, scale, dx, dy)

            # 2. IoU 追蹤 (傳入包含信心值的 list: [x1, y1, x2, y2, conf])
            current_dets = [det[:5] for det in persons if det[4] > 0.4]
            tracked_results = tracker.update(current_dets) # 回傳 {pid: [box, conf]}

            # 3. 派發任務給背景線程
            for pid, (box, conf) in tracked_results.items(): # 修正：這裡解構出 box 和 conf
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = int(max(x2 - x1, y2 - y1) * 1.5)
                
                x_start, y_start = cx - side // 2, cy - side // 2
                x_end, y_end = x_start + side, y_start + side
                
                src_x1, src_y1 = max(0, x_start), max(0, y_start)
                src_x2, src_y2 = min(WIDTH_IN, x_end), min(HEIGHT_IN, y_end)

                if (src_x2 - src_x1) > 10 and (src_y2 - src_y1) > 10:
                    actual_roi = frame[src_y1:src_y2, src_x1:src_x2]
                    t_pad, b_pad = src_y1 - y_start, y_end - src_y2
                    l_pad, r_pad = src_x1 - x_start, x_end - src_x2
                    
                    roi = cv2.copyMakeBorder(actual_roi, t_pad, b_pad, l_pad, r_pad, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    if not pose_task_queue.full():
                        # 修正：確保傳入的變數與 pose_worker 接收端一致
                        pose_task_queue.put((pid, roi.copy(), x_start, y_start, box, conf))

            # 4. 繪製 (從共享結果中安全讀取)
            with results_lock:
                # 使用 list(results.items()) 避免在迭代時字典大小改變導致崩潰
                for pid, res in list(pose_results.items()):
                    if pid not in tracked_results: continue
                    
                    # 執行跌倒邏輯判斷，取得狀態與經過時間
                    status, elapsed = fall_logic.check(pid, res['kpts'], res['box'])
                    
                    kpts_g = res['kpts']
                    #status = res['status']
                    bx = res['box']
                    #elapsed = res.get('elapsed', 0.0)
                    conf_val = res.get('conf', 0.0)

                    # 設定顏色：Normal=綠, CONFIRM=黃, FALL=紅
                    color = (0, 255, 0)
                    if status == "CONFIRM": color = (0, 255, 255)
                    elif status == "FALL": color = (0, 0, 255)

                    # A. 繪製框與 ID 左側信心值
                    cv2.rectangle(frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
#                    cv2.putText(frame, f"{conf_val:.2f}", (int(bx[0]-55), int(bx[1]-10)), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # B. 狀態標籤與秒數
                    label = f"PERSON: {conf_val:.2f} | ID:{pid} [{status}]"
                    if status in ["CONFIRM", "FALL"]:
                        label += f" {elapsed:.1f}s"
                    
                    cv2.putText(frame, label, (int(bx[0]), int(bx[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # C. 繪製骨架
                    # 假設 SKELETON_LINES_16 已定義
                    C_ORG, C_GRN, C_BLU = (0, 165, 255), (0, 255, 0), (255, 255, 0)
                    for p1, p2 in SKELETON_LINES_16:
                        if kpts_g[p1] and kpts_g[p2]:
                            color = C_GRN if (p1==7 and p2==6) else (C_BLU if p1<=6 and p2<=6 else C_ORG)
                            cv2.line(frame, (int(kpts_g[p1][0]), int(kpts_g[p1][1])), (int(kpts_g[p2][0]), int(kpts_g[p2][1])), color, 4)

            # 5. FPS 計算與畫面輸出
            frame_count += 1
            if frame_count % 10 == 0:
                fps_display = 10 / (time.time() - last_time)
                last_time = time.time()
            
            display_frame = cv2.resize(frame, (WIDTH_OUT, HEIGHT_OUT))
            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            out.write(display_frame)

    except KeyboardInterrupt: pass
    finally:
        stream.stop()
        out.release()
