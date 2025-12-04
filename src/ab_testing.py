"""A/B testing framework for comparing ranking systems."""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import config
from src.metrics import MetricsCalculator, RankingMetrics


class Variant(Enum):
    """A/B test variants."""
    CONTROL = "control"  # Baseline
    TREATMENT = "treatment"  # Personalized


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    variant: str
    metrics: RankingMetrics
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "variant": str(self.variant),
            "sample_size": int(self.sample_size)
        }
        # Convert all metric values to native Python types
        metrics_dict = self.metrics.to_dict()
        for key, value in metrics_dict.items():
            result[key] = float(value) if value is not None else 0.0
        return result


@dataclass
class StatisticalSignificance:
    """Statistical significance test results."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": str(self.metric_name),
            "control_mean": float(self.control_mean),
            "treatment_mean": float(self.treatment_mean),
            "p_value": float(self.p_value) if not np.isnan(self.p_value) else None,
            "is_significant": bool(self.is_significant),
            "confidence_interval": [float(self.confidence_interval[0]), float(self.confidence_interval[1])],
            "effect_size": float(self.effect_size) if not np.isnan(self.effect_size) else 0.0,
            "improvement_pct": float(
                (self.treatment_mean - self.control_mean) / self.control_mean * 100
                if self.control_mean > 0 else 0
            )
        }


class ABTest:
    """A/B testing framework."""
    
    def __init__(self, alpha: float = config.AB_TEST_ALPHA):
        """
        Initialize A/B test.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        self.metrics_calculator = MetricsCalculator()
    
    def run_test(
        self,
        control_interactions: List[Dict[str, Any]],
        treatment_interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run A/B test comparing control and treatment.
        
        Args:
            control_interactions: Interactions from control variant
            treatment_interactions: Interactions from treatment variant
            
        Returns:
            Dictionary with test results and significance
        """
        # Calculate metrics for both variants
        control_metrics = self.metrics_calculator.calculate_all_metrics(control_interactions)
        treatment_metrics = self.metrics_calculator.calculate_all_metrics(treatment_interactions)
        
        control_result = ABTestResult(
            variant=Variant.CONTROL.value,
            metrics=control_metrics,
            sample_size=len(control_interactions)
        )
        
        treatment_result = ABTestResult(
            variant=Variant.TREATMENT.value,
            metrics=treatment_metrics,
            sample_size=len(treatment_interactions)
        )
        
        # Perform statistical tests
        significance_tests = self._perform_significance_tests(
            control_interactions,
            treatment_interactions
        )
        
        return {
            "control": control_result.to_dict(),
            "treatment": treatment_result.to_dict(),
            "significance_tests": [test.to_dict() for test in significance_tests],
            "overall_winner": self._determine_winner(significance_tests)
        }
    
    def _perform_significance_tests(
        self,
        control_interactions: List[Dict[str, Any]],
        treatment_interactions: List[Dict[str, Any]]
    ) -> List[StatisticalSignificance]:
        """Perform statistical significance tests on key metrics."""
        tests = []
        
        # Test CTR
        ctr_test = self._test_ctr(control_interactions, treatment_interactions)
        tests.append(ctr_test)
        
        # Test MRR
        mrr_test = self._test_mrr(control_interactions, treatment_interactions)
        tests.append(mrr_test)
        
        # Test NDCG
        ndcg_test = self._test_ndcg(control_interactions, treatment_interactions)
        tests.append(ndcg_test)
        
        # Test engagement rate
        engagement_test = self._test_engagement_rate(control_interactions, treatment_interactions)
        tests.append(engagement_test)
        
        return tests
    
    def _test_ctr(
        self,
        control: List[Dict[str, Any]],
        treatment: List[Dict[str, Any]]
    ) -> StatisticalSignificance:
        """Test click-through rate."""
        control_clicks = self._extract_clicks(control)
        treatment_clicks = self._extract_clicks(treatment)
        
        control_mean = np.mean(control_clicks)
        treatment_mean = np.mean(treatment_clicks)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_clicks, treatment_clicks)
        
        # Confidence interval for difference
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_clicks) / len(control_clicks) +
            np.var(treatment_clicks) / len(treatment_clicks)
        )
        ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_clicks) + np.var(treatment_clicks)) / 2
        )
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalSignificance(
            metric_name="CTR",
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_interval=ci,
            effect_size=effect_size
        )
    
    def _test_mrr(
        self,
        control: List[Dict[str, Any]],
        treatment: List[Dict[str, Any]]
    ) -> StatisticalSignificance:
        """Test mean reciprocal rank."""
        control_mrrs = self._extract_mrrs(control)
        treatment_mrrs = self._extract_mrrs(treatment)
        
        control_mean = np.mean(control_mrrs)
        treatment_mean = np.mean(treatment_mrrs)
        
        t_stat, p_value = stats.ttest_ind(control_mrrs, treatment_mrrs)
        
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_mrrs) / len(control_mrrs) +
            np.var(treatment_mrrs) / len(treatment_mrrs)
        )
        ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)
        
        pooled_std = np.sqrt((np.var(control_mrrs) + np.var(treatment_mrrs)) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalSignificance(
            metric_name="MRR",
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_interval=ci,
            effect_size=effect_size
        )
    
    def _test_ndcg(
        self,
        control: List[Dict[str, Any]],
        treatment: List[Dict[str, Any]]
    ) -> StatisticalSignificance:
        """Test NDCG."""
        control_ndcgs = self._extract_ndcgs(control)
        treatment_ndcgs = self._extract_ndcgs(treatment)
        
        control_mean = np.mean(control_ndcgs)
        treatment_mean = np.mean(treatment_ndcgs)
        
        t_stat, p_value = stats.ttest_ind(control_ndcgs, treatment_ndcgs)
        
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_ndcgs) / len(control_ndcgs) +
            np.var(treatment_ndcgs) / len(treatment_ndcgs)
        )
        ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)
        
        pooled_std = np.sqrt((np.var(control_ndcgs) + np.var(treatment_ndcgs)) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalSignificance(
            metric_name="NDCG",
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_interval=ci,
            effect_size=effect_size
        )
    
    def _test_engagement_rate(
        self,
        control: List[Dict[str, Any]],
        treatment: List[Dict[str, Any]]
    ) -> StatisticalSignificance:
        """Test engagement rate."""
        control_engagements = self._extract_engagements(control)
        treatment_engagements = self._extract_engagements(treatment)
        
        control_mean = np.mean(control_engagements)
        treatment_mean = np.mean(treatment_engagements)
        
        t_stat, p_value = stats.ttest_ind(control_engagements, treatment_engagements)
        
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_engagements) / len(control_engagements) +
            np.var(treatment_engagements) / len(treatment_engagements)
        )
        ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)
        
        pooled_std = np.sqrt(
            (np.var(control_engagements) + np.var(treatment_engagements)) / 2
        )
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalSignificance(
            metric_name="Engagement Rate",
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_interval=ci,
            effect_size=effect_size
        )
    
    def _is_click(self, action: Any) -> bool:
        """Check if action is a click (handles both formats)."""
        if isinstance(action, str):
            return action == "Click"
        elif isinstance(action, dict):
            return action.get("action_type") == "Click"
        return False
    
    def _extract_clicks(self, interactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract click indicators per query."""
        clicks = []
        for interaction in interactions:
            query_clicks = 0
            for action_list in interaction["actions"]:
                if any(self._is_click(a) for a in action_list):
                    query_clicks += 1
            clicks.append(query_clicks / max(len(interaction["actions"]), 1))
        return np.array(clicks)
    
    def _extract_mrrs(self, interactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract MRR per query."""
        mrrs = []
        for interaction in interactions:
            for position, action_list in enumerate(interaction["actions"], 1):
                if any(self._is_click(a) for a in action_list):
                    mrrs.append(1.0 / position)
                    break
            else:
                mrrs.append(0.0)
        return np.array(mrrs)
    
    def _extract_ndcgs(self, interactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract NDCG per query."""
        ndcgs = []
        calculator = MetricsCalculator()
        
        for interaction in interactions:
            relevances = []
            for action_list in interaction["actions"]:
                relevance = calculator._calculate_relevance(action_list)
                relevances.append(relevance)
            
            if not relevances:
                ndcgs.append(0.0)
                continue
            
            # DCG
            dcg = relevances[0]
            for i, rel in enumerate(relevances[1:], 2):
                dcg += rel / np.log2(i + 1)
            
            # IDCG
            sorted_relevances = sorted(relevances, reverse=True)
            idcg = sorted_relevances[0]
            for i, rel in enumerate(sorted_relevances[1:], 2):
                idcg += rel / np.log2(i + 1)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.array(ndcgs)
    
    def _extract_engagements(self, interactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract engagement count per query."""
        engagements = []
        for interaction in interactions:
            engagement_count = 0
            for action_list in interaction["actions"]:
                for action in action_list:
                    action_type = None
                    if isinstance(action, str):
                        action_type = action
                    elif isinstance(action, dict):
                        action_type = action.get("action_type")
                    
                    if action_type in ["Like", "Share", "Bookmark"]:
                        engagement_count += 1
            engagements.append(engagement_count)
        return np.array(engagements)
    
    def _determine_winner(self, tests: List[StatisticalSignificance]) -> str:
        """Determine overall winner based on significance tests."""
        significant_wins = sum(
            1 for test in tests
            if test.is_significant and test.treatment_mean > test.control_mean
        )
        significant_losses = sum(
            1 for test in tests
            if test.is_significant and test.treatment_mean < test.control_mean
        )
        
        if significant_wins > significant_losses:
            return "treatment"
        elif significant_losses > significant_wins:
            return "control"
        else:
            return "inconclusive"


def main():
    """Test A/B testing framework."""
    # Create sample data
    control = [
        {
            "user_id": "user1",
            "actions": [
                ["Click", {"Dwell": {"secs": 30, "nanos": 0}}],
                [],
                ["Click", {"Dwell": {"secs": 20, "nanos": 0}}]
            ]
        }
    ] * 30
    
    treatment = [
        {
            "user_id": "user2",
            "actions": [
                ["Click", {"Dwell": {"secs": 45, "nanos": 0}}, "Like"],
                ["Click", {"Dwell": {"secs": 25, "nanos": 0}}],
                []
            ]
        }
    ] * 30
    
    # Run test
    ab_test = ABTest()
    results = ab_test.run_test(control, treatment)
    
    print("\n=== A/B Test Results ===")
    print(f"\nControl: {results['control']['sample_size']} samples")
    print(f"Treatment: {results['treatment']['sample_size']} samples")
    print(f"\nOverall Winner: {results['overall_winner']}")
    
    print("\n=== Significance Tests ===")
    for test in results['significance_tests']:
        print(f"\n{test['metric_name']}:")
        print(f"  Control: {test['control_mean']:.4f}")
        print(f"  Treatment: {test['treatment_mean']:.4f}")
        print(f"  Improvement: {test['improvement_pct']:.2f}%")
        print(f"  P-value: {test['p_value']:.4f}")
        print(f"  Significant: {test['is_significant']}")


if __name__ == "__main__":
    main()

