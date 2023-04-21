from src.core.evaluators import AccuracyEvaluator, F1ScoreEvaluator, MAEEvaluator, BalancedAccuracyEvaluator

EVALUATORS = {
    "f1": F1ScoreEvaluator,
    "acc": AccuracyEvaluator,
    "bacc": BalancedAccuracyEvaluator,
    "mae": MAEEvaluator,
}


def build(config):
    evaluator = dict()

    for eval_type in config.standards:
        evaluator[eval_type] = EVALUATORS[eval_type]()

    return evaluator
