import json
import sys

def trace_to_text(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    
    # Filter for Complete Events (X) which have duration
    # Also handle metadata for thread naming
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

    for (pid, tid), t_events in threads.items():
        name = thread_names.get((pid, tid), f"Process {pid} Thread {tid}")
        print(f"\n[Thread: {name}]")
        
        # Sort by start time (ts) and then by duration (descending) to ensure parents are before children
        t_events.sort(key=lambda x: (x["ts"], -x["dur"]))
        
        stack = []
        for e in t_events:
            start_offset = (e["ts"] - global_min_ts) / 1000.0  # ms
            duration = e["dur"] / 1000.0  # ms
            
            # Pop stack until we find the parent of this event
            while stack and (stack[-1]["ts"] + stack[-1]["dur"] < e["ts"] + e["dur"]):
                stack.pop()
            
            indent = "  " * len(stack)
            # Clean up name: remove absolute path and keep "func (file.py:line)"
            raw_name = e["name"]
            if " (" in raw_name and raw_name.endswith(")"):
                base_name, path_info = raw_name.rsplit(" (", 1)
                path_info = path_info.rstrip(")")
                if "/" in path_info:
                    file_name = path_info.split("/")[-1]
                else:
                    file_name = path_info
                clean_name = f"{base_name} ({file_name})"
            else:
                clean_name = raw_name
            
            # Token-efficient format: offset duration name
            print(f"{indent}{start_offset:8.1f} {duration:8.1f} {clean_name}")
            
            stack.append(e)

if __name__ == "__main__":
    input_file = "canary_trace.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    trace_to_text(input_file)
