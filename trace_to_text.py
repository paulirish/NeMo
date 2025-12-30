import json
import sys

def trace_to_text(input_file, output_file=None):
    with open(input_file, 'r') as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    
    # Filter for Complete Events (X) which have duration
    thread_names = {}
    duration_events = []
    
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            thread_names[(e["pid"], e["tid"])] = e["args"].get("name")
        elif e.get("ph") == "X":
            duration_events.append(e)

    if not duration_events:
        print("No duration events found.")
        return

    # Group by PID/TID
    threads = {}
    for e in duration_events:
        key = (e["pid"], e["tid"])
        if key not in threads:
            threads[key] = []
        threads[key].append(e)

    # Global start time for relative offsets
    global_min_ts = min(e["ts"] for e in duration_events)

    output_lines = []
    for (pid, tid), t_events in threads.items():
        name = thread_names.get((pid, tid), f"Process {pid} Thread {tid}")
        output_lines.append(f"\n[Thread: {name}]")
        
        # Sort by start time (ts) and then by duration (descending)
        t_events.sort(key=lambda x: (x["ts"], -x["dur"]))
        
        stack = []
        for e in t_events:
            start_offset = (e["ts"] - global_min_ts) / 1000.0  # ms
            duration = e["dur"] / 1000.0  # ms
            
            while stack and (stack[-1]["ts"] + stack[-1]["dur"] < e["ts"] + e["dur"]):
                stack.pop()
            
            indent = "  " * len(stack)
            
            raw_name = e["name"]
            if " (" in raw_name and raw_name.endswith(")"):
                base_name, path_info = raw_name.rsplit(" (", 1)
                path_info = path_info.rstrip(")")
                file_name = path_info.split("/")[-1] if "/" in path_info else path_info
                clean_name = f"{base_name} ({file_name})"
            else:
                clean_name = raw_name
            
            output_lines.append(f"{indent}{start_offset:8.1f} {duration:8.1f} {clean_name}")
            stack.append(e)

    output_content = "\n".join(output_lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_content)
        print(f"Trace text saved to {output_file}")
    else:
        print(output_content)

if __name__ == "__main__":
    input_file = "canary_trace.json"
    output_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    trace_to_text(input_file, output_file)
