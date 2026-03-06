def detect_type(file_path):
    if file_path.endswith((".jpg", ".png")):
        return "xray"
    elif file_path.endswith(".dcm"):
        return "ct_scan"
    elif file_path.endswith(".txt"):
        return "report"
    else:
        return "unknown"
