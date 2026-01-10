import threading

_lock = threading.Lock()
_counters = {}


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels_items):
    if not labels_items:
        return ""
    parts = [f'{k}="{_escape_label_value(str(v))}"' for k, v in labels_items]
    return "{" + ",".join(parts) + "}"


def add(name: str, value: float, **labels) -> None:
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + float(value)


def inc(name: str, **labels) -> None:
    add(name, 1.0, **labels)


def render_prometheus() -> str:
    with _lock:
        items = list(_counters.items())

    items.sort(key=lambda x: (x[0][0], x[0][1]))

    lines = []
    seen_type = set()
    for (name, labels_items), value in items:
        if name not in seen_type:
            lines.append(f"# TYPE {name} counter")
            seen_type.add(name)

        labels_str = _format_labels(labels_items)
        if labels_str:
            lines.append(f"{name}{labels_str} {value}")
        else:
            lines.append(f"{name} {value}")

    return "\n".join(lines) + "\n"
