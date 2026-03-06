COLREG Kernel (MVP)

What this provides (Phase-1: rule structuring only)
- FeatureSchema (pydantic): the input contract for rules.
- Predicate DSL + evaluator (TriBool): executable applicability conditions.
- RuleSpec schema + YAML loader.
- A starter rule library for COLREGs 13–17 + 19 + 9 + 10 (atomic rules, simplified).
- Tests:
  1) schema/lint (YAML validates)
  2) executability (all predicates run)
  3) gold cases (typical scenarios trigger expected rules)

Run tests
1) (optional) create venv
2) pip install -e .[test]
3) pytest -q

Note
- This MVP intentionally keeps actions coarse (TURN_STARBOARD / TURN_PORT / REDUCE_SPEED / MAINTAIN_COURSE_SPEED).
- The rules are structured to support later: scenario generation constraints + scene verification + labeling.
