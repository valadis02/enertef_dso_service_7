"""Extra classifiers for the combination (Liu's own random forest never changes).

xgb_tuned : XGBoost, hyper-parameters tuned with Optuna (TPE, NTRIALS trials) by grouped inner CV on the
            training fold only (nested CV), objective = inner log loss.
xgb_tp    : XGBoost aimed at true positives (finding the open / out-of-service branches, label 0):
            class weight scale_pos_weight is tuned too (<1 up-weights the open class) and the objective is
            the inner average precision of the open class.
tabfm     : Google TabFM v1.0.0 zero-shot, bfloat16 on CPU, TABFM_NEST ensemble members (default 4),
            at most TABFM_MAXN context rows (stratified subsample).
The experiment scripts set GROUPS (a Series indexed like their feature table) before fitting."""
import os

import numpy as np

GROUPS = None
NTRIALS = int(os.environ.get("NTRIALS", "30"))
SEED = 42


def _inner(X, y):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    g = None if GROUPS is None else GROUPS.loc[X.index].values
    if g is not None and len(np.unique(g)) >= 3:
        return list(GroupKFold(min(4, len(np.unique(g)))).split(X, y, g))
    return list(StratifiedKFold(4, shuffle=True, random_state=SEED).split(X, y))


class TunedXGB:
    def __init__(self, tp=False):
        self.tp = tp

    @staticmethod
    def _make(prm):
        import xgboost as xgb
        return xgb.XGBClassifier(**prm, tree_method="hist", random_state=SEED, n_jobs=1, eval_metric="logloss")

    def _space(self, t):
        prm = dict(n_estimators=t.suggest_int("n_estimators", 100, 800, step=50),
                   learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                   max_depth=t.suggest_int("max_depth", 2, 10),
                   min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True),
                   subsample=t.suggest_float("subsample", 0.5, 1.0),
                   colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                   reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
        if self.tp:
            prm["scale_pos_weight"] = t.suggest_float("scale_pos_weight", 0.02, 1.0, log=True)
        return prm

    def fit(self, X, y):
        import optuna
        from sklearn.metrics import average_precision_score, log_loss
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        Xv = np.asarray(X, float); y = np.asarray(y).astype(int); inner = _inner(X, y)

        def obj(t):
            prm = self._space(t); sc = []
            for tr, te in inner:
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                p = self._make(prm).fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
                sc.append(-average_precision_score(y[te] == 0, 1 - p) if self.tp else log_loss(y[te], p, labels=[0, 1]))
            return float(np.mean(sc)) if sc else 0.0

        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(obj, n_trials=NTRIALS)
        self.best_params_ = st.best_params
        self.m = self._make(st.best_params).fit(Xv, y); self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(np.asarray(X, float))


class TabFMZeroShot:
    def __init__(self):
        self.n = int(os.environ.get("TABFM_NEST", "4"))

    def fit(self, X, y):
        import tabfm_lowmem
        from tabfm import TabFMClassifier
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        maxn = int(os.environ.get("TABFM_MAXN", "10000"))
        if len(y) > maxn:
            rng = np.random.default_rng(SEED); keep = []
            for c in np.unique(y):
                ix = np.where(y == c)[0]
                keep.append(rng.choice(ix, max(1, round(maxn * len(ix) / len(y))), replace=False))
            keep = np.sort(np.concatenate(keep)); X, y = X[keep], y[keep]
        self.m = TabFMClassifier(model=tabfm_lowmem.load_cached(), n_estimators=self.n, random_state=SEED)
        self.m.fit(X, y); self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(np.asarray(X, float))


def make(name):
    return {"xgb_tuned": lambda: TunedXGB(False), "xgb_tp": lambda: TunedXGB(True), "tabfm": TabFMZeroShot}[name]()


NAMES = ("xgb_tuned", "xgb_tp", "tabfm")
