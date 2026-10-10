import pytest
import numpy as np

from scipy.stats import cauchy, multivariate_t

from aub_htp.machine_learning.regressor import (
    l_alpha_loss,
    r_alpha_score,
    AlphaStableLinearRegressor,
)


class TestAlphaLossAndScore:
    def test_l_alpha_loss_is_finite_for_perfect_prediction(self):
        """Loss should be finite for perfect prediction."""
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        loss = l_alpha_loss(y, y_pred, alpha=2.0)

        assert np.isfinite(loss)

    def test_l_alpha_loss_is_positive_for_imperfect_prediction(self):
        """Loss should be strictly positive when predictions are not exact."""
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 2.0, 4.0])

        loss = l_alpha_loss(y, y_pred, alpha=2.0)

        assert loss > 0

    def test_r_alpha_score_is_finite_for_perfect_prediction(self):
        """Perfect prediction should produce a finite score."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y.copy()

        score = r_alpha_score(y, y_pred, alpha=2.0)

        assert np.isfinite(score)

    def test_r_alpha_score_for_perfect_prediction_is_at_least_as_good_as_imperfect_prediction(self):
        """Perfect prediction should score at least as well as an imperfect one."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred_perfect = y.copy()
        y_pred_bad = np.array([0.0, 1.0, 2.0, 3.0])

        score_perfect = r_alpha_score(y, y_pred_perfect, alpha=2.0)
        score_bad = r_alpha_score(y, y_pred_bad, alpha=2.0)

        assert score_perfect >= score_bad
    def test_r_alpha_score_is_finite_for_very_bad_prediction(self):
        """Even a very bad prediction should still return a finite score."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([100.0, 100.0, 100.0, 100.0])

        score = r_alpha_score(y, y_pred, alpha=2.0)

        assert np.isfinite(score)

class TestAlphaStableLinearRegressorValidation:
    def test_fit_raises_for_alpha_out_of_range_low(self):
        """fit should reject alpha values less than or equal to zero."""
        model = AlphaStableLinearRegressor(alpha=0.0)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match=r"must be in \(0, 2\]"):
            model.fit(X, y)

    def test_fit_raises_for_alpha_out_of_range_high(self):
        """fit should reject alpha values greater than 2."""
        model = AlphaStableLinearRegressor(alpha=2.5)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match=r"must be in \(0, 2\]"):
            model.fit(X, y)

    def test_predict_raises_if_model_not_fitted(self):
        """predict should fail when called before fit."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0]])

        with pytest.raises(Exception):
            model.predict(X)

    def test_validate_and_reshape_detects_1d_target(self):
        """_validate_and_reshape should mark a 1D target correctly."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        _, y_valid, alpha, y_is_one_dimensional = model._validate_and_reshape(X, y)

        assert y_valid.shape == (4, 1)
        assert alpha == pytest.approx(2.0, rel=1e-12)
        assert y_is_one_dimensional is True

    def test_validate_and_reshape_detects_2d_target(self):
        """_validate_and_reshape should mark a 2D target correctly."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([[2.0], [4.0], [6.0], [8.0]])

        _, y_valid, alpha, y_is_one_dimensional = model._validate_and_reshape(X, y)

        assert y_valid.shape == (4, 1)
        assert alpha == pytest.approx(2.0, rel=1e-12)
        assert y_is_one_dimensional is False

    def test_predict_returns_1d_output_when_y_was_1d(self):
        """predict should return a flat array when fit used a 1D target."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])

        np.random.seed(0)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.ndim == 1
        assert y_pred.shape == (4,)

    def test_predict_returns_2d_output_when_y_was_2d(self):
        """predict should return a 2D array when fit used a 2D target."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([[3.0], [5.0], [7.0], [9.0]])

        np.random.seed(0)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.ndim == 2
        assert y_pred.shape == (4, 1)


class TestAlphaStableLinearRegressorFitPredictUnivariate:
    def test_fit_recovers_simple_linear_relationship_with_normal_X_and_normal_noise(self):
        """fit should approximately recover slope and intercept when X is normal and noise is light-tailed."""
        np.random.seed(123)

        n = 200
        X = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        true_coef = 2.5
        true_intercept = -0.7
        noise = np.random.normal(loc=0.0, scale=0.1, size=n)
        y = true_coef * X[:, 0] + true_intercept + noise

        model = AlphaStableLinearRegressor(alpha=2.0, max_iter=300, tol=1e-4)

        np.random.seed(123)
        model.fit(X, y)

        assert model.coef_.shape == (1, 1)
        assert model.intercept_.shape == (1,)
        assert model.coef_[0, 0] == pytest.approx(true_coef, abs=0.3)
        assert model.intercept_[0] == pytest.approx(true_intercept, abs=0.3)

    def test_predict_has_same_shape_as_target_for_univariate_case(self):
        """Predictions should match the original target shape in the univariate case."""
        np.random.seed(123)

        n = 100
        X = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        noise = np.random.normal(loc=0.0, scale=0.05, size=n)
        y = 1.8 * X[:, 0] + 0.4 + noise

        model = AlphaStableLinearRegressor(alpha=2.0)

        np.random.seed(123)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape

    def test_score_is_high_on_well_specified_normal_data(self):
        """score should be high when the data follow the assumed linear structure closely."""
        np.random.seed(42)

        n = 150
        X = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        noise = np.random.normal(loc=0.0, scale=0.1, size=n)
        y = 3.0 * X[:, 0] - 1.0 + noise

        model = AlphaStableLinearRegressor(alpha=2.0)

        np.random.seed(42)
        model.fit(X, y)
        score = model.score(X, y)

        assert np.isfinite(score)
        assert score > 0.8


class TestAlphaStableLinearRegressorHeavyTailedNoise:
    def test_fit_runs_on_normal_X_with_cauchy_noise(self):
        """fit and predict should work when X is normal and the regression noise is Cauchy."""
        np.random.seed(2024)

        n = 120
        X = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        noise = cauchy.rvs(loc=0.0, scale=0.2, size=n, random_state=2024)
        y = 1.5 * X[:, 0] + 0.2 + noise

        model = AlphaStableLinearRegressor(alpha=1.0, max_iter=300, tol=1e-4)

        np.random.seed(2024)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))

    def test_alpha_one_is_competitive_under_cauchy_noise(self):
        """alpha=1 should perform at least competitively with alpha=2 under Cauchy noise."""
        np.random.seed(999)

        n = 150
        X = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        noise = cauchy.rvs(loc=0.0, scale=0.3, size=n, random_state=999)
        y = 2.0 * X[:, 0] + 1.0 + noise

        model_alpha_1 = AlphaStableLinearRegressor(alpha=1.0, max_iter=300, tol=1e-4)
        model_alpha_2 = AlphaStableLinearRegressor(alpha=2.0, max_iter=300, tol=1e-4)

        np.random.seed(999)
        model_alpha_1.fit(X, y)

        np.random.seed(999)
        model_alpha_2.fit(X, y)

        score_alpha_1 = model_alpha_1.score(X, y)
        score_alpha_2 = model_alpha_2.score(X, y)

        assert np.isfinite(score_alpha_1)
        assert np.isfinite(score_alpha_2)
        assert score_alpha_1 >= score_alpha_2 - 0.1


class TestAlphaStableLinearRegressorMultivariatePredictors:
    def test_fit_with_multivariate_normal_X_and_normal_noise(self):
        """fit should work with multivariate normal predictors and scalar response."""
        np.random.seed(777)

        n = 200
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.4], [0.4, 1.5]])
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

        beta = np.array([1.2, -0.8])
        intercept = 0.3
        noise = np.random.normal(loc=0.0, scale=0.1, size=n)
        y = X @ beta + intercept + noise

        model = AlphaStableLinearRegressor(alpha=2.0)

        np.random.seed(777)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert model.score(X, y) > 0.7

    def test_fit_with_multivariate_normal_X_and_cauchy_noise(self):
        """fit should run on multivariate normal predictors when the noise is heavy-tailed."""
        np.random.seed(2025)

        n = 180
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

        beta = np.array([2.0, -1.0])
        intercept = 0.5
        noise = cauchy.rvs(loc=0.0, scale=0.2, size=n, random_state=2025)
        y = X @ beta + intercept + noise

        model = AlphaStableLinearRegressor(alpha=1.0, max_iter=300, tol=1e-4)

        np.random.seed(2025)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))

    def test_fit_with_multivariate_t_predictors_and_cauchy_noise(self):
        """fit should also run when predictors themselves are heavy-tailed and the noise is Cauchy."""
        np.random.seed(314)

        n = 160
        X = multivariate_t.rvs(
            loc=np.zeros(2),
            shape=np.array([[1.0, 0.3], [0.3, 1.0]]),
            df=1,
            size=n,
            random_state=314,
        )

        beta = np.array([1.0, 2.0])
        intercept = -0.5
        noise = cauchy.rvs(loc=0.0, scale=0.2, size=n, random_state=2718)
        y = X @ beta + intercept + noise

        model = AlphaStableLinearRegressor(alpha=1.0, max_iter=300, tol=1e-4)

        np.random.seed(314)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))


class TestAlphaStableLinearRegressorMultiOutput:
    def test_fit_multioutput_with_multivariate_normal_X_and_normal_noise(self):
        """fit should support multiple outputs when predictors are multivariate normal."""
        np.random.seed(321)

        n = 250
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.25], [0.25, 1.2]])
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

        W = np.array([[2.0, -1.0],
                      [0.5,  3.0]])
        b = np.array([1.0, -2.0])

        noise = np.random.normal(loc=0.0, scale=0.05, size=(n, 2))
        y = X @ W.T + b + noise

        model = AlphaStableLinearRegressor(alpha=2.0, max_iter=300, tol=1e-4)

        np.random.seed(321)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert model.coef_.shape == (2, 2)
        assert model.intercept_.shape == (2,)
        assert y_pred.shape == y.shape

    def test_fit_multioutput_with_multivariate_normal_X_and_heavy_tailed_noise(self):
        """fit should support multiple outputs when responses are corrupted by heavy-tailed noise."""
        np.random.seed(654)

        n = 180
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.1], [0.1, 1.0]])
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

        W = np.array([[1.5, -0.5],
                      [2.0,  1.0]])
        b = np.array([0.2, -1.0])

        noise_1 = cauchy.rvs(loc=0.0, scale=0.15, size=n, random_state=654)
        noise_2 = cauchy.rvs(loc=0.0, scale=0.15, size=n, random_state=655)
        noise = np.column_stack([noise_1, noise_2])

        y = X @ W.T + b + noise

        model = AlphaStableLinearRegressor(alpha=1.0, max_iter=300, tol=1e-4)

        np.random.seed(654)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))


class TestAlphaStableLinearRegressorInternalValidation:
    def test_validate_and_reshape_turns_1d_y_into_column_vector(self):
        """_validate_and_reshape should convert a 1D target into a 2D column vector."""
        model = AlphaStableLinearRegressor(alpha=2.0)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])

        X_valid, y_valid, alpha, y_is_one_dimensional = model._validate_and_reshape(X, y)

        assert X_valid.shape == (3, 1)
        assert y_valid.shape == (3, 1)
        assert alpha == 2.0
        assert y_is_one_dimensional is True

    def test_validate_and_reshape_keeps_2d_y_as_2d(self):
        """_validate_and_reshape should preserve a target that is already 2D."""
        model = AlphaStableLinearRegressor(alpha=1.0)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[2.0], [4.0], [6.0]])

        X_valid, y_valid, alpha, y_is_one_dimensional = model._validate_and_reshape(X, y)

        assert X_valid.shape == (3, 1)
        assert y_valid.shape == (3, 1)
        assert alpha == 1.0
        assert y_is_one_dimensional is False