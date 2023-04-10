from src.core.evaluators import AccuracyEvaluator, R2Evaluator, F1ScoreEvaluator, MAEEvaluator

EVALUATORS = {
    # "r2": R2Evaluator,
    "f1": F1ScoreEvaluator,
    "acc": AccuracyEvaluator,
    "mae": MAEEvaluator,
}


def build(config):
    evaluator = dict()

    for eval_type in config.standards:
        evaluator[eval_type] = EVALUATORS[eval_type]()

    return evaluator
