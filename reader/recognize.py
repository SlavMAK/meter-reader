"""Meter reading: TFLite inference + digit/pointer evaluation.

The evaluation logic (eval_* functions, readout, shift_decimal, replace_n) is a
Python port of AI-on-the-edge-device
(code/components/jomjol_flowcontroll/ClassFlowCNNGeneral.cpp and
ClassFlowPostProcessing.cpp), https://github.com/jomjol/AI-on-the-edge-device
This is a modified version. See LICENSE-AI-on-the-edge.md.
"""
import math

import numpy as np
from PIL import Image

# code/include/defines.h
ANALOG_ERROR = 3
DIGIT_UNCERTAINTY = 0.2
DIGIT_BAND = 3
DIGIT_TRANSITION_AREA_PREDECESSOR = 0.7
DIGIT_TRANSITION_AREA_FORWARD = 9.7


def _round(x):
    """C round(): half away from zero (Python round() is banker's)."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def eval_analog(number, prev):
    """PointerEvalAnalogNew. number: pointer 0..10, prev: evaluated lower pointer or -1."""
    if prev == -1:
        return math.floor(number)
    lo, hi = number - ANALOG_ERROR / 10, number + ANALOG_ERROR / 10
    if math.floor(hi) != math.floor(lo):
        if prev <= ANALOG_ERROR:
            return (math.floor(hi) + 10) % 10
        if prev >= 10 - ANALOG_ERROR:
            return (math.floor(lo) + 10) % 10
    return (math.floor(number) + 10) % 10


def eval_analog_to_digit(number, preceder, eval_pred, transition_start):
    """PointerEvalAnalogToDigitNew: last digit before the first pointer."""
    after = math.floor(number * 10) % 10
    before = (math.floor(number) + 10) % 10
    rounded_up = False
    if after >= 10 - DIGIT_UNCERTAINTY * 10 or (eval_pred <= 4 and after >= 6):
        result = (_round(number) + 10) % 10
        rounded_up = True
        before = result
    else:
        result = (math.trunc(number) + 10) % 10
    if eval_pred >= 6 and (preceder > transition_start or preceder <= 0.2) and rounded_up:
        result = (before + 10 - 1) % 10
    return result


def eval_hybrid(number, pred_number, eval_pred, analog_pred=False, transition_start=9.2):
    """PointerEvalHybridNew: a digit judged by the (lower) digit/pointer to its right."""
    after = math.floor(number * 10) % 10
    before = (math.floor(number) + 10) % 10
    if eval_pred < 0:
        return _round(number * 100) // 100  # 9.995+ gives 10, caller turns it into N
    if analog_pred:
        return eval_analog_to_digit(number, pred_number, eval_pred, transition_start)
    if DIGIT_TRANSITION_AREA_PREDECESSOR <= pred_number <= 10 - DIGIT_TRANSITION_AREA_PREDECESSOR:
        if after <= DIGIT_BAND or after >= 10 - DIGIT_BAND:
            return (_round(number) + 10) % 10
        return (math.trunc(number) + 10) % 10
    if eval_pred <= 1:  # the lower digit has already rolled over
        return (before + 1) % 10 if after > 5 else before
    if DIGIT_TRANSITION_AREA_FORWARD >= pred_number or after >= 4:
        return before
    return (before - 1 + 10) % 10


def readout(digits, analogs, transition_start=9.2):
    """Raw reading like "0123.4N6". Lists are most significant first; digit < 0 means N."""
    analog_str, prev = "", -1
    for v in reversed(analogs):
        prev = eval_analog(v, prev)
        analog_str = str(prev) + analog_str

    digit_str = ""
    if digits:
        last = digits[-1]
        if 0 <= last < 10:
            if analogs:
                prev = eval_hybrid(last, analogs[0], prev, True, transition_start)
            else:
                prev = eval_hybrid(last, prev, prev)
            digit_str = str(prev) if 0 <= prev < 10 else "N"
        else:
            digit_str = "N"
        for i in range(len(digits) - 2, -1, -1):
            if 0 <= digits[i] < 10:
                prev = eval_hybrid(digits[i], digits[i + 1], prev)
                digit_str = str(prev) + digit_str
            else:
                prev = -1
                digit_str = "N" + digit_str
    return digit_str + ("." + analog_str if analogs else "")


def shift_decimal(s, shift):
    """ShiftDecimal: move the decimal point; -3 turns "12345" into "12.345"."""
    if shift == 0:
        return s
    pos = s.find(".")
    if pos < 0:
        pos = len(s)
    else:
        s = s[:pos] + s[pos + 1:]
    new = pos + shift
    if new <= 0:
        return "0." + "0" * -new + s
    if new > len(s):
        return s + "0" * (new - len(s))
    return s[:new] + "." + s[new:]


def replace_n(raw, pre):
    """ErsetzteN: fill unreadable positions (N) with digits of the previous value."""
    int_part, dot, frac = raw.partition(".")
    p_int, _, p_frac = f"{pre:.{len(frac)}f}".partition(".")
    p_int = p_int.rjust(len(int_part), "0")[-len(int_part):] if int_part else ""
    out = "".join(p if c == "N" else c for c, p in zip(int_part, p_int))
    return out + dot + "".join(p if c == "N" else c for c, p in zip(frac, p_frac))


def evaluate(raw, pre, pre_ts, now, max_flow_m3h):
    """Plausibility checks on a raw reading. Returns (value, error); value is None if rejected."""
    if not raw.replace(".", ""):
        return None, "Не размечены области цифр и стрелок"
    s = raw
    if "N" in s:
        if pre is None:
            return None, f"Не распознаны цифры ({raw}), предыдущего значения нет"
        s = replace_n(s, pre)
    decimals = len(s.partition(".")[2])
    value = round(float(s), decimals)
    if pre is not None:
        if value < pre:
            return None, f"Показание уменьшилось: {raw} < {pre}"
        hours = max((now - pre_ts) / 3600, 1 / 60)
        if max_flow_m3h and (value - pre) / hours > max_flow_m3h:
            return None, f"Слишком большой расход: {raw}, было {pre}"
    return value, None


class Model:
    """A TFLite model from AI-on-the-edge; kind is detected by its output size like the firmware does."""

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        self.n = int(self.out["shape"][-1])
        if self.n not in (2, 10, 11, 100):
            raise ValueError(f"{path}: модель с {self.n} выходами не поддерживается")

    def __call__(self, img, ccw=False):
        """Returns 0..<10 (e.g. 3.7), or -1 for "not readable"."""
        x = np.asarray(img.convert("RGB").resize((self.w, self.h), Image.BILINEAR), dtype=np.float32)
        self.it.set_tensor(self.inp["index"], x[None])
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0]
        if self.n == 2:  # pointer: (sin, cos) of the angle
            v = (math.atan2(y[0], y[1]) / (2 * math.pi) + 2) % 1 * 10
        elif self.n == 10:  # dig-cont: weighted between the two best neighbours
            num = int(y.argmax())
            plus, minus, val = float(y[(num + 1) % 10]), float(y[(num - 1) % 10]), float(y[num])
            v = num + plus / (plus + val) if plus > minus else num - minus / (minus + val)
        elif self.n == 11:  # dig-class11: 0..9 or N
            c = int(y.argmax())
            return float(c) if c < 10 else -1.0
        else:  # 100 classes: 0.0 .. 9.9
            v = int(y.argmax()) / 10
        if ccw:
            v = 10 - v
        return v % 10 if math.isfinite(v) else -1.0
