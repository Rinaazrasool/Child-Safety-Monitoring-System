# ==============================================================================
#      Child Safety AI - Optimized Local Version v3.12 (Final Syntax Fix)
# ==============================================================================
import flask
import cv2
import numpy as np
import time
import os
import traceback
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from collections import deque
from twilio.rest import Client # Using Twilio

app = flask.Flask(__name__)

# --- 1. CONFIGURATION ---
ESP32_STREAM_URL = "http://10.158.147.194:81/stream" # <<< Make sure this is correct!
RISK_THRESHOLD = 70
WARNING_THRESHOLD = 40
YOLO_FRAME_SKIP = 3
AGE_EMOTION_FRAME_SKIP = 30
HAZARD_ZONES = { "pool_edge": (0, 240, 640, 480), "unsafe_stairs": (50, 50, 100, 100) }
DANGEROUS_OBJECTS = [
    'knife', 'scissors', 'oven', 'microwave', 'toaster', 'sink', 'car', 'motorcycle',
    'bus', 'truck', 'boat', 'fire hydrant', 'fork', 'spoon', 'bottle', 'cup', 'wine glass',
]
PROXIMITY_THRESHOLD = 70

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.environ.get('TWILIO_FROM_NUMBER')
PARENT_PHONE_NUMBER = os.environ.get('PARENT_PHONE_NUMBER')
SMS_ALERT_COOLDOWN = 120
last_sms_time = 0
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and PARENT_PHONE_NUMBER:
    try: twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN); print("✅ Twilio client initialized.")
    except Exception as e: print(f"⚠️ WARNING: Failed Twilio init: {e}. SMS disabled.")
else: print("⚠️ WARNING: Twilio credentials incomplete. SMS disabled.")

# --- 2. LOAD AI MODELS ---
model_status = {"yolo": False, "age": False, "emotion": False}
try: object_model = YOLO('yolov8n.pt'); print("✅ YOLO loaded."); model_status["yolo"] = True
except Exception as e: print(f"❌ YOLO Error: {e}"); object_model = None
try: age_model_name='nateraw/vit-age-classifier'; age_processor=ViTImageProcessor.from_pretrained(age_model_name); age_model=ViTForImageClassification.from_pretrained(age_model_name); print(f"✅ Age model loaded."); model_status["age"]=True
except Exception as e: print(f"❌ Age Model Error: {e}"); age_model,age_processor=None,None
try: emotion_model_name='trpakov/vit-face-expression'; emotion_processor=ViTImageProcessor.from_pretrained(emotion_model_name); emotion_model=ViTForImageClassification.from_pretrained(emotion_model_name); print(f"✅ Emotion model loaded."); model_status["emotion"]=True
except Exception as e: print(f"❌ Emotion Model Error: {e}"); emotion_model,emotion_processor=None,None
ALL_MODELS_OPERATIONAL = all(model_status.values())

# --- 3. GLOBAL STATE & HELPERS ---
current_state = { "risk_level": 0, "status_text": "Initializing...", "is_live": False, "alert_history": [], "fps": 0.0, "models_loaded": model_status }
person_data_cache = {}
MAX_DETECTION_MEMORY = 5
last_detections = deque(maxlen=MAX_DETECTION_MEMORY)
frame_count_fps = 0; start_time_fps = time.time()
consecutive_frame_read_failures = 0

def check_overlap(box, zone): x_left=max(box[0],zone[0]); y_top=max(box[1],zone[1]); x_right=min(box[2],zone[2]); y_bottom=min(box[3],zone[3]); return not (x_right<x_left or y_bottom<y_top)
def add_alert_to_history(event_text): global current_state; timestamp=time.strftime("%H:%M:%S"); alert_entry={"time":timestamp,"event":event_text}; current_state["alert_history"].insert(0,alert_entry); current_state["alert_history"]=current_state["alert_history"][:10]; print(f"🚨 ALERT LOGGED: {timestamp} - {event_text}")

# --- CORRECTED classify_age ---
def classify_age(crop):
    if age_model is None: return 'unknown'
    try: # Start of try block
        pil_image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        inputs=age_processor(images=pil_image,return_tensors="pt")
        with torch.no_grad():
            outputs=age_model(**inputs)
        logits=outputs.logits
        label=age_model.config.id2label[logits.argmax(-1).item()]
        return 'child' if label in ['0-2','3-9','10-19'] else 'adult'
    except Exception as e: # Correctly indented except block
        print(f"⚠️ Age cls error: {e}")
        return 'unknown'

# --- CORRECTED classify_emotion ---
def classify_emotion(crop):
    if emotion_model is None: return 'unknown'
    try: # Start of try block
        pil_image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        inputs=emotion_processor(images=pil_image,return_tensors="pt")
        with torch.no_grad():
            outputs=emotion_model(**inputs)
        logits=outputs.logits
        label=emotion_model.config.id2label[logits.argmax(-1).item()]
        if label in ['fear','sadness','disgust','anger']: return 'distress'
        else: return 'casual'
    except Exception as e: # Correctly indented except block
        print(f"⚠️ Emotion cls error: {e}")
        return 'unknown'

def is_near(box1, box2, threshold): c1_x=(box1[0]+box1[2])/2; c1_y=(box1[1]+box1[3])/2; c2_x=(box2[0]+box2[2])/2; c2_y=(box2[1]+box2[3])/2; distance=np.sqrt((c1_x-c2_x)**2 + (c1_y-c2_y)**2); return distance < threshold
def send_sms_alert(message_body):
    global last_sms_time
    if not twilio_client: return
    current_time = time.time()
    if (current_time - last_sms_time) > SMS_ALERT_COOLDOWN:
        try: print(f"📲 Sending SMS Alert to {PARENT_PHONE_NUMBER}..."); message = twilio_client.messages.create(body=f"🚨 Child Safety Alert: {message_body}", from_=TWILIO_FROM_NUMBER, to=PARENT_PHONE_NUMBER); print(f"✅ SMS sent (SID: {message.sid})."); last_sms_time = current_time
        except Exception as e: print(f"❌ ERROR sending SMS: {e}")

# --- 5. CORE AI PROCESSING FUNCTION ---
# (process_frames function remains unchanged from v3.9/v3.10)
def process_frames():
    global current_state, person_data_cache, last_detections
    global frame_count_fps, start_time_fps, consecutive_frame_read_failures

    if not model_status["yolo"]: print("❌ CRITICAL: YOLO model failed."); current_state.update({"status_text": "YOLO Load Error", "is_live": False}); return

    frame_counter = 0

    while True: # Outer loop
        cap = None
        try: # Connect
            print(f"Attempting connection: {ESP32_STREAM_URL}..."); cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            if not cap.isOpened(): raise ConnectionError("Failed to open stream.")
        except Exception as e: # Handle fail
            print(f"❌ VideoCapture Error: {e}. Retrying..."); current_state.update({"is_live": False, "status_text": "Stream Connect Error"})
            last_detections.clear(); person_data_cache.clear();
            if cap: cap.release(); time.sleep(5); continue

        print("✅ Stream connected."); current_state["is_live"] = True
        start_time_fps = time.time(); frame_count_fps = 0; consecutive_frame_read_failures = 0

        while True: # Inner loop
            frame = None
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_frame_read_failures += 1; print(f"⚠️ Frame grab failed ({consecutive_frame_read_failures}).")
                    if consecutive_frame_read_failures > 10: raise ConnectionError("Too many frame read failures.")
                    time.sleep(0.1); continue
                else: consecutive_frame_read_failures = 0

                frame_count_fps += 1; frame_counter += 1
                run_yolo = frame_counter % YOLO_FRAME_SKIP == 0
                run_age_emotion = frame_counter % AGE_EMOTION_FRAME_SKIP == 0

                temp_detections = []

                if run_yolo and object_model:
                    yolo_results = object_model.track(frame, persist=True, classes=[0] + [object_model.names.index(obj) for obj in DANGEROUS_OBJECTS if obj in object_model.names], verbose=False, conf=0.3)
                    children_found=[]; adult_present=False; risks_found=[]
                    processed_ids=set()
                    if yolo_results and yolo_results[0].boxes:
                        for box in yolo_results[0].boxes:
                            class_id=int(box.cls[0]); c_name=object_model.names[class_id]; conf=float(box.conf[0])
                            coords=[int(c) for c in box.xyxy[0]]; x1,y1,x2,y2=coords
                            det={"class":c_name,"box":coords,"conf":conf,"type":"other","track_id":None,"emotion":None}
                            if c_name in DANGEROUS_OBJECTS: det["type"]="risk"; risks_found.append(coords)
                            elif c_name=='person':
                                tid=int(box.id[0]) if box.id is not None else None;
                                if tid is None: continue; processed_ids.add(tid); det["track_id"]=tid
                                crop=frame[y1:y2,x1:x2] if y2>y1 and x2>x1 else None
                                age=person_data_cache.get(tid,{}).get('age','unknown'); emo=person_data_cache.get(tid,{}).get('emotion','unknown')
                                if run_age_emotion and crop is not None and crop.size>0:
                                    if age=='unknown' and model_status["age"]: age=classify_age(crop)
                                    if model_status["emotion"]: emo=classify_emotion(crop)
                                    if tid not in person_data_cache: person_data_cache[tid]={}
                                    person_data_cache[tid]['age']=age; person_data_cache[tid]['emotion']=emo; person_data_cache[tid]['last_seen']=frame_counter
                                elif tid in person_data_cache: person_data_cache[tid]['last_seen']=frame_counter
                                det["emotion"]=emo
                                if age=='child': det["type"]="child"; children_found.append({"box":coords,"status":"Safe","emotion":emo})
                                elif age=='adult': det["type"]="adult"; adult_present=True
                                else: det["type"]="person_unknown"
                            temp_detections.append(det)
                    stale_ids=[tid for tid,data in person_data_cache.items() if frame_counter-data.get('last_seen',0)>AGE_EMOTION_FRAME_SKIP*10]
                    for tid in stale_ids:
                        if tid in person_data_cache: print(f"Removing stale track ID: {tid}"); del person_data_cache[tid]

                    max_risk=0; status="Safe"
                    if not children_found: status="No children detected"
                    is_unsupervised = not adult_present
                    for child in children_found:
                        in_zone=False; near_obj=False; r_name=""
                        for name,zone in HAZARD_ZONES.items():
                            if check_overlap(child["box"],zone): in_zone=True; r_name=name; break
                        if not in_zone:
                            for r_box in risks_found:
                                if is_near(child["box"],r_box,PROXIMITY_THRESHOLD): near_obj=True;
                                for d in temp_detections:
                                    if d["type"]=="risk" and d["box"]==r_box: r_name=d["class"]; break
                                break
                        is_distressed = child["emotion"]=='distress'
                        current_risk=0; current_status="Safe"
                        is_immediate_danger = in_zone or near_obj or is_distressed
                        if is_unsupervised:
                            if is_immediate_danger:
                                current_risk=100
                                if is_distressed: current_status="CRITICAL: Unsupervised child appears distressed!"
                                elif in_zone: current_status=f"CRITICAL: Unsupervised child in {r_name}!"
                                else: current_status=f"CRITICAL: Unsupervised child near {r_name}!"
                            else: current_risk=45; current_status="Warning: Child Unsupervised"
                        else:
                            if is_immediate_danger:
                                current_risk=10
                                if is_distressed: current_status="Info: Supervised child appears distressed."
                                elif in_zone: current_status=f"Info: Supervised child near {r_name}."
                                else: current_status=f"Info: Supervised child near {r_name}."
                            else: current_risk=10; current_status="Safe: Supervised"
                        child["status"]=current_status
                        if current_risk > max_risk: max_risk=current_risk; status=current_status

                    should_log = (max_risk>=WARNING_THRESHOLD and current_state["risk_level"]<WARNING_THRESHOLD) or (max_risk>=RISK_THRESHOLD and current_state["risk_level"]<RISK_THRESHOLD)
                    should_trigger_critical = max_risk>=RISK_THRESHOLD and current_state["risk_level"]<RISK_THRESHOLD
                    if should_log: add_alert_to_history(status)
                    if should_trigger_critical: send_sms_alert(status)
                    current_state.update({"risk_level":max_risk,"status_text":status})
                    last_detections.appendleft(temp_detections)

                # --- Draw boxes and stream ---
                draw_frame = frame.copy()
                current_draw_detections = last_detections[0] if last_detections else []
                # ... (Drawing logic) ...
                for name,(x1,y1,x2,y2) in HAZARD_ZONES.items(): cv2.rectangle(draw_frame,(x1,y1),(x2,y2),(0,255,255),2); cv2.putText(draw_frame,name.upper(),(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                for det in current_draw_detections:
                    x1,y1,x2,y2=det["box"]; label=det["class"]; conf=det.get("conf",0); type=det.get("type","other")
                    tid=det.get("track_id"); emo=person_data_cache.get(tid,{}).get('emotion','') if tid else ''
                    color=(200,200,200); text=f"{label} {conf*100:.0f}%"
                    if type=="child": color=(0,255,0); text=f"CHILD(ID:{tid}) {emo}";
                    if type=="child" and emo=='distress': color=(0,0,255)
                    elif type=="adult": color=(255,100,0); text=f"ADULT(ID:{tid})"
                    elif type=="risk": color=(0,0,255); text=f"RISK: {label} {conf*100:.0f}%"
                    elif type=="person_unknown": color=(128,0,128); text=f"PERSON?(ID:{tid})"
                    cv2.rectangle(draw_frame,(x1,y1),(x2,y2),color,2)
                    (w,h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                    t_bg_y1=max(y1-h-10,0); t_bg_y2=max(y1,0)
                    cv2.rectangle(draw_frame,(x1,t_bg_y1),(x1+w,t_bg_y2),color,-1); cv2.putText(draw_frame,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

                elapsed_time=time.time()-start_time_fps
                if elapsed_time>=1.0: fps=frame_count_fps/elapsed_time; current_state["fps"]=round(fps,1); start_time_fps=time.time(); frame_count_fps=0

                status_color=(0,255,0); risk=current_state["risk_level"]; status=current_state["status_text"]
                if risk>=RISK_THRESHOLD: status_color=(0,0,255)
                elif risk>=WARNING_THRESHOLD: status_color=(0,165,255)
                cv2.putText(draw_frame,f"STATUS: {status}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)
                cv2.putText(draw_frame,f"Risk: {risk}%",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)
                cv2.putText(draw_frame,f"FPS: {current_state['fps']:.1f}",(frame.shape[1]-100,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

                (flag, encodedImage) = cv2.imencode(".jpg", draw_frame)
                if not flag: print("⚠️ JPEG encoding failed."); continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            # --- CORRECTED except blocks ---
            except ConnectionError as e:
                print(f"Stream connection error: {e}. Reconnecting...")
                current_state.update({"is_live": False, "status_text": "Reconnecting..."})
                last_detections.clear()
                person_data_cache.clear()
                if cap:
                    cap.release()
                    cap = None
                time.sleep(2)
                break # Break inner loop to reconnect

            except Exception as e:
                print(f"❌ Unexpected Error in processing loop: {e}")
                traceback.print_exc()
                time.sleep(1) # Continue inner loop after general error
            # --- END CORRECTED except blocks ---

        # Code after inner loop breaks
        if cap: cap.release()
        print("Inner loop exited. Waiting before reconnect...")
        time.sleep(3)
    # End of outer loop
    print("Outer loop exited.")
# --- END process_frames ---


# --- 6. FLASK ROUTES ---
@app.route("/")
def index(): return flask.render_template("index.html")
@app.route("/video_feed")
def video_feed(): headers={'Cache-Control':'no-cache,no-store,must-revalidate','Pragma':'no-cache','Expires':'0'}; return flask.Response(process_frames(),mimetype="multipart/x-mixed-replace; boundary=frame",headers=headers)
@app.route("/get_status")
def get_status(): current_state["models_loaded"]=model_status; return flask.jsonify(current_state)

# --- 7. RUN THE APP ---
if __name__ == "__main__":
    print("\n--- Initializing Child Safety Monitor (Local v3.11 Syntax Fix) ---") # Version update
    if not ALL_MODELS_OPERATIONAL:
        print("--- ⚠️ Startup Warning ---");
        if not model_status["yolo"]: print("- YOLO model failed.")
        if not model_status["age"]: print("- Age model failed.")
        if not model_status["emotion"]: print("- Emotion model failed.")
        print("- AI features limited."); print("-------------------------\n")
    else:
        print("--- ✅ Starting Server ---");
        print("Models: Local YOLO, Age, Emotion")
        print(f"YOLO:{YOLO_FRAME_SKIP}f, Age/Emo:{AGE_EMOTION_FRAME_SKIP}f")
        if twilio_client: print(f"SMS Alerts Enabled: Sending to {PARENT_PHONE_NUMBER}")
        else: print("SMS Alerts Disabled: Check Twilio credentials/setup.")
        print("Connect: http://YOUR_IP:5000 or http://127.0.0.1:5000"); print("---------------------------\n")

    try: from waitress import serve; print("Running Waitress server."); serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError: print("Waitress not found. Using Flask dev server."); app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    