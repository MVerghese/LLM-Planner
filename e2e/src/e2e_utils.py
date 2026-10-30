from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm

def load_images(image_paths):
    """Load images from a list of file paths."""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    return images


def encode_image(img):
    """Encode image to base64 string"""
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        numpy_img = np.array(img)
    else:
        numpy_img = img
        
    # Convert RGB to BGR for OpenCV
    if len(numpy_img.shape) == 3 and numpy_img.shape[2] == 3:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        
    _, JPEG = cv2.imencode('.jpeg', numpy_img)
    return base64.b64encode(JPEG).decode('utf-8')

def precompute_dataset_embeddings(knn_set: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Precompute the embeddings for the dataset using the specified model.

    Args:
        knn_set (pd.DataFrame): The dataset containing the text to embed.
        model_name (str): The name of the model to use for embedding.

    Returns:
        pd.DataFrame: The original dataset with an additional column for embeddings.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(knn_set["task_instr"].tolist(), show_progress_bar=True)
    # Cast embeddings to float
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    knn_set["embeddings"] = embeddings.tolist()
    return knn_set

def update_knn_set_with_embeddings(knn_set_path: str, model_name: str) -> pd.DataFrame:
    """
    Load the KNN dataset, compute embeddings, and save the updated dataset.

    Args:
        knn_set_path (str): The file path to the KNN dataset.
        model_name (str): The name of the model to use for embedding.

    Returns:
        pd.DataFrame: The updated KNN dataset with embeddings.
    """
    knn_set = pd.read_pickle(knn_set_path)
    knn_set = precompute_dataset_embeddings(knn_set, model_name)
    knn_set.to_pickle(knn_set_path)
    return knn_set

def explore_knn_set(knn_set_path: str) -> None:
    """
    Explore the KNN dataset by printing basic statistics and a sample of the data.

    Args:
        knn_set (pd.DataFrame): The KNN dataset to explore.
    """
    knn_set = pd.read_pickle(knn_set_path)
    print("KNN Headers:")
    print(knn_set.columns.tolist())
    print("KNN Set Overview:")
    print(knn_set.describe())
    print("\nSample Data:")
    print(knn_set.sample(5))
    import pdb; pdb.set_trace()


def _json_serializer(obj):
    """Convert numpy and other non-JSON-serializable types for json.dump."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_knn_set_task_and_traj(knn_set: pd.DataFrame, out_path: str) -> None:
    """
    Dump the "task_instr" and "gold_traj" columns from the knn_set to a JSON file.

    Args:
        knn_set (pd.DataFrame): The KNN dataset (e.g. from pd.read_pickle).
        out_path (str): Path for the output JSON file.
    """
    subset = knn_set[["task_instr", "gold_traj"]].copy()
    records = subset.to_dict(orient="records")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_serializer, ensure_ascii=False)


def _object_id_to_name(object_id: str) -> str:
    """Return the object name (characters before the first '|') from an objectId string."""
    if "|" in object_id:
        return object_id.split("|", 1)[0]
    return object_id


def api_actions_to_text_steps(api_actions: List[Dict[str, Any]]) -> List[str]:
    """
    Parse api_action dictionaries into text steps: action, plus object name from
    objectId if present, plus object name from receptacleObjectId if present.
    Object names are taken as the characters before the first '|' in each id.

    Args:
        api_actions: List of api_action dicts (e.g. from extract_api_actions_from_plan_json).

    Returns:
        List of text step strings, e.g. ["LookDown", "PickupObject Potato", "PutObject Potato Fridge"].
    """
    steps = []
    for d in api_actions:
        action = d.get("action", "")
        parts = [action]
        if "objectId" in d:
            parts.append(_object_id_to_name(d["objectId"]))
        if "receptacleObjectId" in d:
            parts.append(_object_id_to_name(d["receptacleObjectId"]))
        steps.append(" ".join(parts))
    return steps


def _is_movement_step(step: str) -> bool:
    """True if step is a movement step (starts with Look, Move, or Rotate)."""
    if not step.strip():
        return False
    action = step.split()[0]
    return (
        action.startswith("Look")
        or action.startswith("Move")
        or action.startswith("Rotate")
    )


def _get_last_object_from_step(step: str) -> str:
    """Return the last object (last token after the action) in a step, or empty string."""
    parts = step.split()
    if len(parts) >= 2:
        return parts[-1]
    return ""


def collapse_movement_to_navigation(steps: List[str]) -> List[str]:
    """
    Replace runs of movement steps (Look*, Move*, Rotate*) with a single "Navigation <object>"
    step, where <object> is taken from the next non-movement step (e.g. "PickupObject Knife" -> "Navigation Knife").
    Trailing movement steps with no following non-movement step become "Navigation".

    Args:
        steps: List of text step strings (e.g. from api_actions_to_text_steps).

    Returns:
        List of steps with movement runs collapsed to one Navigation step each.
    """
    result = []
    i = 0
    while i < len(steps):
        if _is_movement_step(steps[i]):
            # Skip all consecutive movement steps
            while i < len(steps) and _is_movement_step(steps[i]):
                i += 1
            # Target object from the next non-movement step
            if i < len(steps):
                target = _get_last_object_from_step(steps[i])
                result.append("Navigation " + target if target else "Navigation")
            else:
                result.append("Navigation")
        else:
            result.append(steps[i])
            i += 1
    return result

def fix_countertop_navigation_steps(steps: List[str]) -> List[str]:
    """
    For every "Navigation countertop" step, insert a "Navigation <next_item>"
    step before the next interaction (such as PickupObject or PutObject) that targets
    an object distinct from countertop. Intended to reduce ambiguity when navigating to
    a large countertop by enforcing explicit navigation to the next object.

    Args:
        steps: List of step strings.

    Returns:
        List of steps with additional necessary navigation steps after countertop navigation.
    """
    result = []
    i = 0
    current_location = ""
    while i < len(steps):
        step = steps[i]
        if step.startswith("Navigation"):
            current_location = step.split()[-1].lower()
        if current_location == "countertop" and "countertop" not in step.lower():
            result.append(f"Navigation {step.split()[-1]}")
        result.append(step)
        i += 1
    return result

def fix_navigation_steps(steps: List[str]) -> List[str]:
    """
    Fix the navigation steps by adding the current location to the step.
    """
    result = []
    i = 0
    current_location = ""
    while i < len(steps):
        step = steps[i]
        if step.startswith("Navigation"):
            current_location = step.split()[-1].lower()
        else:
            object_name = step.split()[-1]
            if current_location != object_name.lower():
                result.append(f"Navigation {object_name}")
        result.append(step)
        i += 1
    return result    

def extract_api_actions_from_plan_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Extract the "api_action" dictionaries from the "low_actions" list inside the
    "plan" object of a JSON file (e.g. ALFRED traj_data.json).

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of api_action dictionaries, in the same order as in low_actions.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    plan = data.get("plan")
    if plan is None:
        return []
    low_actions = plan.get("low_actions", [])
    return [item["api_action"] for item in low_actions if "api_action" in item]


def load_json_into_knn_set(knn_set: pd.DataFrame, json_path: str) -> pd.DataFrame:
    """
    Replace the "task_instr" and "gold_traj" columns in knn_set with values from a JSON file.
    The JSON must be a list of objects with "task_instr" and "gold_traj" keys, in the same
    order as the rows of knn_set (e.g. as produced by dump_knn_set_task_and_traj).

    Args:
        knn_set (pd.DataFrame): The KNN dataset to update.
        json_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: The knn_set with task_instr and gold_traj replaced. Modify in place
            and return for convenience; caller can save with knn_set.to_pickle(...).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) != len(knn_set):
        raise ValueError(
            f"JSON has {len(records)} records but knn_set has {len(knn_set)} rows; "
            "lengths must match."
        )
    # Build a mapping from task_instr to the record's gold_traj
    traj_map = {r["task_instr"]: r["gold_traj"] for r in records}
    # Preserve the existing order of task_instr in knn_set
    knn_set["gold_traj"] = [traj_map[ti] if ti in traj_map else None for ti in knn_set["task_instr"]]
    # Also update task_instr column to match the records, but preserve order in knn_set
    # (If task_instrs don't match exactly or are shuffled, this preserves current knn_set order)
    # Optionally: Uncomment the next line to force update them with ordered records
    knn_set["task_instr"] = [r["task_instr"] for r in records]
    return knn_set


KNN_SET_JSON_COLUMNS = ["task", "task_instr", "step_instr", "gold_traj", "vis_objs"]


def _knn_json_field(r: Dict[str, Any], key: str, default: Any) -> Any:
    """Value for ``key`` in record ``r``, or ``default`` if missing or JSON null."""
    v = r.get(key, default)
    return default if v is None else v


def knn_set_from_json(json_path: str) -> pd.DataFrame:
    """
    Build a new KNN DataFrame from a JSON file. The file must contain a JSON array of objects.

    Columns are ``task``, ``task_instr``, ``step_instr``, ``gold_traj``, and ``vis_objs``.
    A key missing from an object or present with JSON null is filled with defaults:
    ``""``, ``""``, ``""``, ``[]``, ``{}``.

    Args:
        json_path: Path to the JSON file (list of records).

    Returns:
        pd.DataFrame with exactly the columns in ``KNN_SET_JSON_COLUMNS``.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"JSON root must be a list, got {type(records).__name__}")
    rows = []
    for r in records:
        if not isinstance(r, dict):
            raise TypeError(f"Each JSON element must be an object, got {type(r).__name__}")
        rows.append(
            {
                "task": _knn_json_field(r, "task", ""),
                "task_instr": _knn_json_field(r, "task_instr", ""),
                "step_instr": _knn_json_field(r, "step_instr", ""),
                "gold_traj": _knn_json_field(r, "gold_traj", []),
                "vis_objs": _knn_json_field(r, "vis_objs", {}),
            }
        )
    return pd.DataFrame(rows, columns=KNN_SET_JSON_COLUMNS)


ALFRED_JSON_ROOT = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/alfred/data/json_2.1.0"


def search_json_files_for_phrase(
    query: str,
    root_dir: Union[str, Path] = ALFRED_JSON_ROOT,
    case_sensitive: bool = False,
) -> List[Tuple[Path, List[int]]]:
    """
    Search all JSON files under a root directory for a query phrase.

    Args:
        query: The phrase to search for.
        root_dir: Directory to search recursively; defaults to ALFRED json_2.1.0 path.
        case_sensitive: If False, search is case-insensitive.

    Returns:
        List of (file_path, line_numbers) for each file that contains the phrase.
        Line numbers are 1-based.
    """
    root = Path(root_dir)
    if not root.is_dir():
        return []
    matches = []
    search_text = query if case_sensitive else query.lower()
    for path in root.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        content = text if case_sensitive else text.lower()
        if search_text not in content:
            continue
        lines = text.splitlines()
        line_nos = [
            i + 1
            for i, line in enumerate(lines)
            if search_text in (line if case_sensitive else line.lower())
        ]
        matches.append((path, line_nos))
    return matches

def generate_gold_trajs_from_json(task_and_traj_json_path: str, save_path: str) -> None:
    """
    Generate gold trajectories from a JSON file.

    Args:
        task_and_traj_json_path: Path to the JSON file.
        save_path: Path to save the generated gold trajectories.
    """
    with open(task_and_traj_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for record in tqdm(records):
        task_instr = record["task_instr"]
        matches = search_json_files_for_phrase(task_instr)
        json_files = [path for path, _ in matches]
        # Get the first json file with the word "traj_data" in the path
        traj_data_json_file = next((path for path in json_files if "traj_data" in path.name), None)
        if traj_data_json_file is None:
            continue
        api_actions = extract_api_actions_from_plan_json(traj_data_json_file)
        steps = api_actions_to_text_steps(api_actions)
        collapsed_steps = collapse_movement_to_navigation(steps)
        fixed_steps = fix_navigation_steps(collapsed_steps)
        # Convert each collapsed step to required stringified tuple and wrap in a list
        formatted_steps = []
        for step in fixed_steps:
            # Assumes step is in the form "Action Obj1 [Obj2]"
            parts = step.split()
            if len(parts) == 2:
                # Example: "OpenObject Fridge"
                action, obj1 = parts
                tuple_str = f"('{action}', '{obj1}')"
            elif len(parts) == 3:
                # Example: "PutObject Apple Fridge"
                action, obj1, obj2 = parts
                tuple_str = f"('{action}', '{obj1}', '{obj2}')"
            elif len(parts) == 1:
                # Just the action
                action = parts[0]
                tuple_str = f"('{action}')"
            else:
                # Fallback: join all into tuple, lowercase all except the first
                action = parts[0]
                objects = [p for p in parts[1:]]
                args = "', '".join([action] + objects)
                tuple_str = f"('{args}')"
            formatted_steps.append([tuple_str])
        fixed_steps = formatted_steps
        record["gold_traj"] = fixed_steps
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_serializer, ensure_ascii=False)

        

def main():
    knn_set_path = "src/knn_set.pkl"
    knn_set_save_path = "src/knn_set_fruit.pkl"

    # knn_set_path = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/prompt_iteration/knn_sets/knn_set_iteration_3.pkl"
    model_name = "paraphrase-MiniLM-L6-v2"


    # updated_knn_set = update_knn_set_with_embeddings(knn_set_path, model_name)
    # print("Updated KNN Set:")
    # print(updated_knn_set)
    # explore_knn_set(knn_set_path)
    # 1/0
    # knn_set = pd.read_pickle(knn_set_path)
    # dump_knn_set_task_and_traj(knn_set, "src/knn_set_task_and_traj_base.json")
    # Default root (ALFRED json_2.1.0)
    # for path, line_nos in search_json_files_for_phrase("Slice the potato using the knife.  Put the knife in the fridge with the potato in the pan. "):
    #     print(path, line_nos)
    
    # traj_data_path = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/alfred/data/json_2.1.0/train/pick_and_place_with_movable_recep-PotatoSliced-Pot-Fridge-20/trial_T20190906_185011_982418/traj_data.json"
    # api_actions = extract_api_actions_from_plan_json(traj_data_path)
    # steps = api_actions_to_text_steps(api_actions)
    # collapsed_steps = collapse_movement_to_navigation(steps)
    # print(collapsed_steps)
    # generate_gold_trajs_from_json("src/knn_set_task_and_traj.json", "src/knn_set_task_and_traj_gold_trajs_with_navigation.json")
    # print("Gold trajectories generated and saved to src/knn_set_task_and_traj_gold_trajs_with_navigation.json")

    # knn_set = pd.read_pickle(knn_set_path)
    # knn_set = load_json_into_knn_set(knn_set, "src/knn_set_task_and_traj_gold_trajs_with_navigation.json")
    # knn_set.to_pickle(knn_set_save_path)
    knn_set = knn_set_from_json("src/knn_set_task_and_traj_fruit.json")
    print("KNN set from JSON:")
    print(knn_set)
    knn_set.to_pickle(knn_set_save_path)
    update_knn_set_with_embeddings(knn_set_save_path, model_name)

    1/0


if __name__ == "__main__":
    main()