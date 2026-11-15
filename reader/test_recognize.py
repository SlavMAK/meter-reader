"""Run: python test_recognize.py  (vectors copied from AI-on-the-edge unit tests)."""
from recognize import (eval_analog, eval_analog_to_digit, eval_hybrid, evaluate, readout,
                       replace_n, shift_decimal)


def value(digits, analogs, shift=0):
    raw = shift_decimal(readout(digits, analogs), shift)
    v, err = evaluate(raw, None, 0, 0, 0)
    assert err is None, err
    return f"{v:.{len(raw.partition('.')[2])}f}"


def test_eval():
    # test_cnnflowcontroll.cpp
    assert [eval_analog(5.2, 3), eval_analog(5.2, 9), eval_analog(4.4, 9), eval_analog(4.5, 0)] == [5, 4, 4, 4]
    cases = [((5.2, 0, -1), 5), ((5.3, 0, -1), 5), ((5.7, 0, -1, False, 9.2), 5), ((5.8, 8.0, 8, False, 8.0), 6),
             ((5.7, 0.4, 1), 6), ((5.3, 0.7, 1), 5), ((5.3, 0.1, 1), 5), ((5.2, 9.8, 9, False, 9.0), 4),
             ((5.7, 9.6, 9), 5), ((4.5, 9.6, 0), 4), ((4.5, 9.6, 9), 4), ((4.5, 9.5, 9), 4), ((8.6, 9.8, 9), 8),
             ((1.8, 9.0, 9), 2), ((4.6, 6.0, 6), 4)]
    for args, want in cases:
        assert eval_hybrid(*args) == want, (args, want)
    # test_PointerEvalAnalogToDigitNew.cpp (ContinuousLastDigit=false cases)
    cases = [((4.8, 8.0, 8, 9.2), 5), ((9.6, 6.8, 6, 9.2), 9), ((4.6, 6.2, 6, 9.2), 4), ((6.8, 8.6, 6, 9.2), 7),
             ((6.8, 1.0, 1, 9.2), 7), ((3.9, 9.7, 9, 9.2), 3), ((4.0, 9.1, 9, 9.2), 4), ((9.8, 0.1, 9, 9.2), 9),
             ((5.9, 9.4, 9, 9.2), 5), ((1.8, 7.8, 7, 7.7), 1), ((8.0, 9.88, 9, 9.2), 8), ((7.8, 6.3, 6, 9.2), 8)]
    for args, want in cases:
        assert eval_analog_to_digit(*args) == want, (args, want)


def test_flow():
    # test_flowpostprocessing.cpp, test_doFlowPP / test_doFlowPP1
    cases = [([1.2, 6.7], [9.5, 8.4], "16.98"), ([3.0, 7.0, 6.0, 5.0, 2.5, 9.6], [6.4], "376529.6"),
             ([1.1, 6.0, 7.0, 7.0, 3.0, 4.6], [6.2], "167734.6"), ([5.0, 8.6], [9.8, 6.7, 8.9, 8.6, 9.8], "58.96889"),
             ([2.9, 7.0, 6.8, 9.9, 8.0, 3.9], [9.7], "377083.9"), ([1.1, 9.0, 4.0], [6.1, 2.6, 6.25, 9.7], "194.6259"),
             ([1.1, 9.0, 4.0], [8.1, 2.6, 6.25, 9.7], "194.8259"), ([1.1, 9.0, 4.0], [9.1, 2.6, 6.25, 9.7], "194.9259"),
             ([1.0, 9.0, 9.0], [7.1, 4.8, 8.3], "199.748"), ([0.0, 2.9, 3.0, 2.9, 3.5, 9.5], [], "33339"),
             ([9.9, 2.8, 2.9, 2.9, 3.7, 9.7], [], "33339"), ([0.0, 9.9, 6.8, 9.9, 3.7, 0.8, 6.9, 8.7], [], "704178"),
             ([9.9, 6.8, 1.1, 4.7, 2.7, 6.0, 9.0, 2.8], [], "7153692")]
    for digits, analogs, want in cases:
        assert value(digits, analogs) == want, (digits, analogs, value(digits, analogs), want)
    # 8 rollers, the 3 red ones are litres
    assert value([0, 0, 1, 2, 3, 4, 5, 6], [], shift=-3) == "123.456"


def test_postprocess():
    assert evaluate("", None, 0, 0, 3)[0] is None  # no ROIs drawn yet
    assert shift_decimal("123", 2) == "12300" and shift_decimal("12", -3) == "0.012"
    assert replace_n("01N3.4N6", 123.456) == "0123.456"
    assert replace_n("0N.29", 10.29) == "00.29"
    assert evaluate("123.4N6", None, 0, 0, 3)[0] is None  # N without a previous value
    assert evaluate("123.4N6", 123.456, 0, 3600, 3) == (123.456, None)
    assert evaluate("123.455", 123.456, 0, 3600, 3)[0] is None  # went down
    assert evaluate("223.456", 123.456, 0, 3600, 3)[0] is None  # +100 m³ in an hour
    assert evaluate("125.456", 123.456, 0, 3600, 3) == (125.456, None)


if __name__ == "__main__":
    test_eval()
    test_flow()
    test_postprocess()
    print("ok")
