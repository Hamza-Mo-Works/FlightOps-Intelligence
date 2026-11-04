"""Load source declarations and notebook expressions without running setup or writes."""
import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATHS = (
    ROOT / "databricks_apps/Conversational-AI-Frontend",
    ROOT / "databricks_apps/flightops-intelligence_2026_10_25-16_57/Conversational-AI-Frontend",
)


def source_path(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing project source: {path}")
    return path


def app_path(name):
    for folder in APP_PATHS:
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f"Cannot find {name} in either known frontend directory")


def tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))


def literal_value(path, name):
    for node in tree(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Source assignment {name!r} not found in {path}")


def declarations(path):
    """Load function bodies, safe constants and library imports; omit top-level I/O."""
    namespace = {"__name__": "flightops_source_under_test"}
    module = tree(path)
    imports = []
    allowed = {"json", "os", "re", "time", "datetime", "decimal", "functools", "typing", "zoneinfo", "math", "pandas", "numpy"}
    for node in module.body:
        if isinstance(node, ast.Import) and all(item.name.split('.')[0] in allowed for item in node.names):
            imports.append(copy.deepcopy(node))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in allowed:
            imports.append(copy.deepcopy(node))
    if imports:
        exec(compile(ast.fix_missing_locations(ast.Module(body=imports, type_ignores=[])), str(path), "exec"), namespace)
    # Constant expressions may combine existing constants, but may not perform I/O.
    pending = [node for node in module.body if isinstance(node, ast.Assign)]
    allowed_calls = {"date", "tuple", "list", "set", "dict", "range", "len"}
    for _ in range(len(pending) + 1):
        progressed = False
        for node in pending[:]:
            if any(isinstance(child, ast.Call) and not (isinstance(child.func, ast.Name) and child.func.id in allowed_calls)
                   for child in ast.walk(node.value)):
                pending.remove(node)
                continue
            try:
                exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])), str(path), "exec"), namespace)
            except (NameError, AttributeError, TypeError):
                continue
            pending.remove(node)
            progressed = True
        if not progressed:
            break
    definitions = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node = copy.deepcopy(node)
            node.decorator_list = []
            definitions.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def assignments(path, names, namespace):
    """Execute only named assignments; reads, table writes and notebook displays are omitted."""
    parsed = tree(path)
    for name in names:
        candidates = [node for node in ast.walk(parsed) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        if not candidates:
            raise AssertionError(f"Assignment {name} not found in {path}")
        node = copy.deepcopy(candidates[0])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def semantic_records():
    source = source_path("05_Business_Semantic_Layer.py")
    rows = literal_value(source, "metric_definitions")
    keys = ("metric_id", "metric_name", "business_definition", "gold_table", "numerator_column",
            "denominator_column", "aggregation_rule", "unit", "synonyms")
    return {row[0]: dict(zip(keys, row)) for row in rows}


def backend_namespace():
    ns = declarations(app_path("flightops_backend.py"))
    specs = semantic_records()
    ns["load_semantic_metrics"] = lambda: specs
    ns["valid_airports"] = lambda: {"ATL", "JFK", "LAX", "ORD", "DFW"}
    return ns


def predictor_namespace():
    return declarations(app_path("flightops_predictor.py"))


def weather_namespace():
    ns = declarations(app_path("flightops_v2_predictor.py"))
    v1 = predictor_namespace()
    ns["PredictionUnavailable"] = v1["PredictionUnavailable"]
    ns["model_features"] = v1["model_features"]
    return ns


def standard_plan(domain="OPS", intent="metric_calculation", metrics=None, **changes):
    chosen = {"OPS": "M001", "AIRPORT": "M007", "CARRIER": "M009"}
    plan = {"intent": intent, "metric_ids": metrics or [chosen[domain]],
            "start_date": "2025-01-01", "end_date": "2025-12-31"}
    plan.update(changes)
    return plan


def base_reference():
    return dict(month=11, route="JFK_LAX", distance_miles=2475.0, day_of_week=2,
                is_weekend=0, airline_prior_flights=100.0, airline_prior_delay_rate=.2,
                airport_prior_flights=200.0, airport_prior_delay_rate=.25,
                route_prior_flights=80.0, route_prior_delay_rate=.3)



def v2_case(weather, observations=None, probability=.3):
    """Capture the frame passed to the classifier by the real V2 predictor body."""
    from datetime import date, time
    import numpy as np
    v1 = predictor_namespace()
    v1["run_sql"] = lambda *args: [base_reference()]
    arguments = (date(2025, 11, 5), "AA", "JFK", "LAX", time(10), time(13))
    frame = v1["model_features"](*arguments)
    captured = {}
    class CaptureClassifier:
        def predict_proba(self, value):
            captured["frame"] = value.copy()
            return np.array([[1 - probability, probability]])
    manifest = {"threshold": .2157, "model_uri": "models:/unit-test-classifier",
                "category_levels": {column: [str(frame.iloc[0][column])] for column in weather["CATEGORICAL"]}}
    weather["model_features"] = lambda *args: frame.copy()
    weather["_origin_timezone"] = lambda airport: "America/New_York"
    weather["_weather_at_cutoff"] = lambda *args: observations or {}
    weather["_load_v2_release"] = lambda: (CaptureClassifier(), manifest)
    return arguments, captured, manifest
