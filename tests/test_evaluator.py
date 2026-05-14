import pytest

from banana_ripeness.evaluator import Evaluator, MetricsReport, RIPENESS_STAGES


@pytest.fixture
def ev():
    return Evaluator()


class TestPerfectPredictions:
    def test_perfect_precision_recall_f1(self, ev):
        labels = list(range(4)) * 10
        report = ev.evaluate(labels, labels)
        for m in report.per_class:
            assert m.precision == pytest.approx(1.0)
            assert m.recall == pytest.approx(1.0)
            assert m.f1 == pytest.approx(1.0)

    def test_macro_f1_is_one_when_perfect(self, ev):
        labels = list(range(4)) * 5
        report = ev.evaluate(labels, labels)
        assert report.macro_f1 == pytest.approx(1.0)


class TestAllWrong:
    def test_zero_precision_recall_f1_all_wrong(self, ev):
        # Predict class 1 for everything that is class 0
        preds = [1, 1, 1, 1]
        trues = [0, 0, 0, 0]
        report = ev.evaluate(preds, trues)
        # class 0: tp=0, fp=0, fn=4 → precision=0, recall=0
        class0 = next(m for m in report.per_class if m.stage == "unripe")
        assert class0.precision == pytest.approx(0.0)
        assert class0.recall == pytest.approx(0.0)
        assert class0.f1 == pytest.approx(0.0)


class TestKnownMetrics:
    def test_known_precision_recall(self, ev):
        # preds=[0,0,0,1], trues=[0,0,1,1]
        # class 0: tp=2, fp=1 (pred 0 but true 1), fn=0 → precision=2/3, recall=2/2=1.0
        preds = [0, 0, 0, 1]
        trues = [0, 0, 1, 1]
        report = ev.evaluate(preds, trues)
        c0 = next(m for m in report.per_class if m.stage == "unripe")
        assert c0.precision == pytest.approx(2 / 3)
        assert c0.recall == pytest.approx(1.0)

    def test_support_counts_true_instances(self, ev):
        preds = [0, 0, 1, 2]
        trues = [0, 1, 1, 3]
        report = ev.evaluate(preds, trues)
        supports = {m.stage: m.support for m in report.per_class}
        assert supports["unripe"] == 1       # one true 0
        assert supports["nearly-ripe"] == 2  # two true 1s
        assert supports["ripe"] == 0
        assert supports["overripe"] == 1


class TestMissingClass:
    def test_missing_class_in_predictions_f1_zero(self, ev):
        # Class 2 (ripe) never predicted
        preds = [0, 0, 1, 1, 3, 3]
        trues = [0, 2, 1, 2, 3, 2]
        report = ev.evaluate(preds, trues)
        ripe = next(m for m in report.per_class if m.stage == "ripe")
        assert ripe.precision == pytest.approx(0.0)
        assert ripe.recall == pytest.approx(0.0)
        assert ripe.f1 == pytest.approx(0.0)


class TestStringLabels:
    def test_accepts_stage_strings(self, ev):
        preds = ["unripe", "ripe", "overripe", "nearly-ripe"]
        trues = ["unripe", "ripe", "overripe", "nearly-ripe"]
        report = ev.evaluate(preds, trues)
        assert report.macro_f1 == pytest.approx(1.0)

    def test_unknown_stage_string_raises(self, ev):
        with pytest.raises(ValueError, match="Unknown ripeness stage"):
            ev.evaluate(["green"], ["unripe"])

    def test_invalid_index_raises(self, ev):
        with pytest.raises(ValueError, match="out of range"):
            ev.evaluate([99], [0])


class TestEdgeCases:
    def test_mismatched_lengths_raises(self, ev):
        with pytest.raises(ValueError, match="same length"):
            ev.evaluate([0, 1], [0])

    def test_empty_inputs_raises(self, ev):
        with pytest.raises(ValueError, match="must not be empty"):
            ev.evaluate([], [])

    def test_report_has_all_four_stages(self, ev):
        labels = list(range(4))
        report = ev.evaluate(labels, labels)
        stages = {m.stage for m in report.per_class}
        assert stages == set(RIPENESS_STAGES)

    def test_str_representation(self, ev):
        labels = list(range(4)) * 3
        report = ev.evaluate(labels, labels)
        text = str(report)
        assert "Macro F1" in text
        for stage in RIPENESS_STAGES:
            assert stage in text
