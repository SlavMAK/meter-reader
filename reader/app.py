"""Meter reader: takes snapshots from ESPHome ESP32-CAMs, recognizes them, publishes to Home Assistant via MQTT."""
import base64
import hmac
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import paho.mqtt.client as mqtt
from PIL import Image

import recognize

DATA = Path(os.environ.get("DATA_DIR", "/data"))
MODELS = Path(os.environ.get("MODELS_DIR", "/app/models"))
STATIC = Path(__file__).with_name("static")
CONFIG, STATE = DATA / "config.json", DATA / "state.json"
UI_PASSWORD = os.environ.get("UI_PASSWORD", "")
PREFIX = os.environ.get("MQTT_DISCOVERY_PREFIX", "homeassistant")
AVAIL = "meter-reader/status"

DEFAULTS = {
    "name": "", "host": "", "flash_brightness": 80, "flash_delay_s": 2.0, "interval_min": 60, "rotate": 0.0,
    "decimal_shift": 0, "max_flow_m3h": 3.0, "transition_start": 9.2,
    "digit_model": "dig-cont_0900_s3_q.tflite", "analog_model": "ana-cont_1500_s2_q.tflite",
    "digits": [], "analogs": [],
}
MQTT_DEFAULTS = {"host": "", "port": 1883, "username": "", "password": ""}
LIMITS = {"flash_brightness": (0, 255, int), "flash_delay_s": (0, 30, float), "interval_min": (1, 44640, int),
          "rotate": (-180, 180, float), "decimal_shift": (-9, 9, int), "max_flow_m3h": (0, 1000, float),
          "transition_start": (0, 10, float)}

log = logging.getLogger("meter-reader")
# ponytail: one lock for all cameras and models (readings run one at a time); per-meter locks if it ever gets slow
LOCK = threading.Lock()
_models = {}


def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written previous value


def model_names():
    return sorted(p.name for p in MODELS.glob("*.tflite"))


def clean_meter(m):
    """Validate a meter coming from the UI (trust boundary: id ends up in file names and MQTT topics)."""
    m = {**DEFAULTS, **m}
    if not re.fullmatch(r"[a-z0-9_]{1,32}", str(m.get("id", ""))):
        raise ValueError("id: 1-32 символа a-z, 0-9, _")
    out = {"id": m["id"], "name": str(m["name"]), "host": str(m["host"]).strip()}
    if out["host"] and not re.fullmatch(r"[A-Za-z0-9.\-]+(:\d+)?", out["host"]):
        raise ValueError(f"{m['id']}: host — IP или имя хоста, например 192.168.1.50")
    for k, (lo, hi, typ) in LIMITS.items():
        v = typ(float(m[k]))
        if not lo <= v <= hi:
            raise ValueError(f"{m['id']}: {k} должно быть в диапазоне {lo}..{hi}")
        out[k] = v
    names = model_names()
    for k in ("digit_model", "analog_model"):
        if m[k] not in names:
            raise ValueError(f"{m['id']}: нет модели {m[k]}")
        out[k] = m[k]
    out["digits"] = [{k: int(r[k]) for k in "xywh"} for r in m["digits"]]
    out["analogs"] = [{**{k: int(r[k]) for k in "xywh"}, "ccw": bool(r.get("ccw"))} for r in m["analogs"]]
    if any(r["w"] < 2 or r["h"] < 2 for r in out["digits"] + out["analogs"]):
        raise ValueError(f"{m['id']}: слишком маленькая область")
    return out


def clean_mqtt(new, old):
    """A missing password keeps the saved one: the UI never receives it back."""
    m = {**MQTT_DEFAULTS, **old, **{k: v for k, v in new.items() if k in MQTT_DEFAULTS}}
    host, port = str(m["host"]).strip(), int(m["port"])
    if host and not re.fullmatch(r"[A-Za-z0-9.\-]+", host):
        raise ValueError("MQTT: сервер — IP или имя хоста, например 192.168.1.10")
    if not 1 <= port <= 65535:
        raise ValueError("MQTT: порт 1..65535")
    return {"host": host, "port": port, "username": str(m["username"]), "password": str(m["password"])}


def clean_config(cfg, old_mqtt):
    meters = [clean_meter(m) for m in cfg.get("meters", [])]
    ids = [m["id"] for m in meters]
    if len(set(ids)) != len(ids):
        raise ValueError("id счётчиков должны быть уникальны")
    return {"mqtt": clean_mqtt(cfg.get("mqtt", {}), old_mqtt), "meters": meters}


def public(cfg):
    mq_cfg = {k: v for k, v in cfg["mqtt"].items() if k != "password"}
    return {**cfg, "mqtt": {**mq_cfg, "password_set": bool(cfg["mqtt"]["password"])}}


# --- camera (ESPHome web_server REST API + esp32_camera_web_server snapshot on :8081) ---

def http(url, method="GET", timeout=15):
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def capture(m):
    if not m["host"]:
        raise RuntimeError("не задан host камеры")
    snapshot = f"http://{m['host'].split(':')[0]}:8081/"
    flash = m["flash_brightness"] > 0

    def flash_on():  # also restarts the camera's 15 s flash watchdog
        if flash:
            http(f"http://{m['host']}/light/Flash/turn_on?brightness={m['flash_brightness']}", "POST")

    try:
        flash_on()
        time.sleep(m["flash_delay_s"])  # let auto exposure settle under the flash
        try:
            http(snapshot)  # discard: the buffered frame may predate the flash
        except (OSError, HTTPException):
            pass
        for attempt in range(1, 4):  # a weak Wi-Fi link cuts transfers off midway
            try:
                return http(snapshot)
            except (OSError, HTTPException) as e:
                if attempt == 3:
                    raise
                log.warning("%s: snapshot attempt %d failed: %s", m["id"], attempt, e)
                flash_on()
    finally:
        if flash:
            try:
                http(f"http://{m['host']}/light/Flash/turn_off", "POST")
            except (OSError, HTTPException) as e:
                log.warning("%s: flash off failed (%s); the camera switches it off by itself", m["id"], e)


def open_image(jpeg, rotate):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return img.rotate(rotate, resample=Image.BICUBIC) if rotate else img


def run_models(jpeg, m):
    img = open_image(jpeg, m["rotate"])

    def run(name, rois):
        if not rois:
            return []
        if name not in _models:
            _models[name] = recognize.Model(MODELS / name)
        return [_models[name](img.crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])), r.get("ccw", False))
                for r in rois]

    digits, analogs = run(m["digit_model"], m["digits"]), run(m["analog_model"], m["analogs"])
    raw = recognize.shift_decimal(recognize.readout(digits, analogs, m["transition_start"]), m["decimal_shift"])
    return {"digits": [round(v, 2) for v in digits], "analogs": [round(v, 2) for v in analogs], "raw": raw}


# --- state + MQTT ---

state = load_json(STATE, {}) if DATA.exists() else {}
config = {"mqtt": dict(MQTT_DEFAULTS), "meters": []}
mq, mqtt_status = None, "не настроен"


def pub(topic, payload):
    if mq:
        mq.publish(topic, payload, qos=1, retain=True)  # qos 1 is queued while disconnected


def read_meter(m):
    """Capture, recognize, check, store and publish one meter. Errors end up in the state, not raised."""
    with LOCK:
        st = state.setdefault(m["id"], {"value": None, "ts": 0})
        now = time.time()
        st.update(time=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), raw=None)
        try:
            jpeg = capture(m)
            (DATA / f"{m['id']}.jpg").write_bytes(jpeg)
            res = run_models(jpeg, m)
            value, error = recognize.evaluate(res["raw"], st["value"], st["ts"], now, m["max_flow_m3h"])
            st.update(raw=res["raw"], error=error)
            if value is not None:
                st.update(value=value, ts=now)
        except Exception as e:
            log.exception("%s: reading failed", m["id"])
            st["error"] = f"{type(e).__name__}: {e}"
        log.info("%s: raw=%s value=%s error=%s", m["id"], st["raw"], st["value"], st["error"])
        save_json(STATE, state)
        publish_state(m["id"])
        return dict(st)


def publish_state(mid):
    st = state.get(mid, {})
    pub(f"meter-reader/{mid}/state", json.dumps(
        {k: st.get(k) for k in ("value", "raw", "error", "time")}, ensure_ascii=False))


def discovery_topics(mid):
    return {k: f"{PREFIX}/sensor/meter_reader_{mid}/{k}/config" for k in ("value", "raw", "error")}


def publish_discovery(m):
    mid, topic = m["id"], f"meter-reader/{m['id']}/state"
    dev = {"identifiers": [f"meter_reader_{mid}"], "name": m["name"] or mid, "manufacturer": "meter-reader",
           "model": "ESP32-CAM"}
    common = {"state_topic": topic, "availability_topic": AVAIL, "device": dev}
    payloads = {
        "value": {"name": "Показание", "value_template": "{{ value_json.value }}", "device_class": "water",
                  "state_class": "total_increasing", "unit_of_measurement": "m³", "json_attributes_topic": topic},
        "raw": {"name": "Распознано", "value_template": "{{ value_json.raw }}", "entity_category": "diagnostic",
                "icon": "mdi:eye"},
        "error": {"name": "Ошибка", "value_template": "{{ value_json.error or 'OK' }}",
                  "entity_category": "diagnostic", "icon": "mdi:alert-circle-outline"},
    }
    for k, t in discovery_topics(mid).items():
        pub(t, json.dumps({**common, **payloads[k], "unique_id": f"meter_reader_{mid}_{k}"}, ensure_ascii=False))


def set_status(client, text):
    global mqtt_status
    if client is mq:  # ignore late callbacks of a replaced client
        mqtt_status = text
        log.info("MQTT: %s", text)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        return set_status(client, f"ошибка: {reason_code}")
    set_status(client, "подключено")
    pub(AVAIL, "online")
    for m in config["meters"]:
        publish_discovery(m)
        publish_state(m["id"])


def on_disconnect(client, userdata, flags, reason_code, properties):
    if mqtt_status == "подключено":  # keep a more specific error such as "Not authorized"
        set_status(client, f"нет соединения: {reason_code}")


def mqtt_start(c):
    """(Re)connect with new settings; on_connect republishes everything."""
    global mq, mqtt_status
    old, mq = mq, None
    if old:
        old.publish(AVAIL, "offline", qos=1, retain=True)
        old.disconnect()
        old.loop_stop()
    if not c["host"]:
        mqtt_status = "не настроен"
        return
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # random client id: two instances must not kick each other
    if c["username"]:
        client.username_pw_set(c["username"], c["password"])
    client.will_set(AVAIL, "offline", qos=1, retain=True)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_connect_fail = lambda cl, u: set_status(cl, f"не удаётся подключиться к {c['host']}:{c['port']}")
    mq, mqtt_status = client, f"подключение к {c['host']}:{c['port']}…"
    client.connect_async(c["host"], c["port"])
    client.loop_start()


def set_config(new):
    global config
    for mid in {m["id"] for m in config["meters"]} - {m["id"] for m in new["meters"]}:
        for t in discovery_topics(mid).values():
            pub(t, "")  # removes the entity from HA
    mqtt_changed = new["mqtt"] != config["mqtt"]
    config = new
    save_json(CONFIG, new)
    if mqtt_changed:
        mqtt_start(new["mqtt"])
    else:
        for m in new["meters"]:
            publish_discovery(m)


def scheduler():
    last = {}
    while True:
        for m in config["meters"]:
            ready = m["host"] and (m["digits"] or m["analogs"])  # nothing to read before the ROIs are drawn
            if ready and time.time() - last.get(m["id"], 0) >= m["interval_min"] * 60:
                last[m["id"]] = time.time()
                read_meter(m)
        time.sleep(20)


# --- web UI / API ---

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def send(self, code, body=b"", ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if code == 401:
            self.send_header("WWW-Authenticate", 'Basic realm="meter-reader"')
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def authorized(self):
        if not UI_PASSWORD:
            return True
        auth = self.headers.get("Authorization", "")
        try:
            password = base64.b64decode(auth.removeprefix("Basic ")).decode().partition(":")[2]
        except ValueError:
            return False
        return hmac.compare_digest(password.encode(), UI_PASSWORD.encode())

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def route(self, method):
        if not self.authorized():
            return self.send(401, {"error": "auth"})
        url = urlsplit(self.path)
        p = url.path.strip("/").split("/")
        try:
            if method == "GET" and p == [""]:
                return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p == ["api", "config"]:
                if method == "PUT":
                    set_config(clean_config(self.body(), config["mqtt"]))
                return self.send(200, public(config))
            if p == ["api", "models"]:
                return self.send(200, model_names())
            if p == ["api", "status"]:
                return self.send(200, {"mqtt": mqtt_status, "meters": state})
            if len(p) == 4 and p[:2] == ["api", "meters"]:
                return self.meter_action(method, p[2], p[3], url)
            self.send(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as e:
            self.send(400, {"error": str(e)})
        except (OSError, HTTPException) as e:  # camera unreachable, transfer cut off
            self.send(502, {"error": f"{type(e).__name__}: {e}"})

    def meter_action(self, method, mid, action, url):
        if not re.fullmatch(r"[a-z0-9_]{1,32}", mid):
            raise ValueError("bad id")
        image = DATA / f"{mid}.jpg"
        if method == "GET" and action == "image":
            if not image.exists():
                return self.send(404, {"error": "нет снимка"})
            rotate = float(parse_qs(url.query).get("rotate", ["0"])[0])
            buf = io.BytesIO()
            open_image(image.read_bytes(), rotate).save(buf, "JPEG", quality=92)
            return self.send(200, buf.getvalue(), "image/jpeg")
        if method != "POST":
            return self.send(405, {"error": "method"})
        if action == "read":
            m = next((m for m in config["meters"] if m["id"] == mid), None)
            return self.send(200, read_meter(m)) if m else self.send(404, {"error": "счётчик не сохранён"})
        m = clean_meter({**self.body(), "id": mid})  # unsaved settings from the editor
        with LOCK:
            if action == "snapshot":
                image.write_bytes(capture(m))
                return self.send(200, {"ok": True})
            if action == "test":
                if not image.exists():
                    return self.send(404, {"error": "нет снимка"})
                return self.send(200, run_models(image.read_bytes(), m))
        self.send(404, {"error": "not found"})


def main():
    global config
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    DATA.mkdir(parents=True, exist_ok=True)
    config = clean_config(load_json(CONFIG, {}), MQTT_DEFAULTS)
    mqtt_start(config["mqtt"])
    threading.Thread(target=scheduler, daemon=True).start()
    port = int(os.environ.get("PORT", 8000))
    log.info("UI on http://0.0.0.0:%d", port)
    ThreadingHTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
