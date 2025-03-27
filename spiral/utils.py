def reformat_llama_prompt(text):
    """
    Remove the "Cutting Knowledge Date" and "Today Date" lines from the text. \n
    Add a newline before the "<|start_header_id|>user<|end_header_id|>" marker.
    """
    marker_user = "<|start_header_id|>user<|end_header_id|>"
    marker_assistant = "<|start_header_id|>assistant<|end_header_id|>"
    
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Cutting Knowledge Date:"):
            i += 1
            continue
        elif lines[i].startswith("Today Date:"):
            i += 1
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        else:
            if marker_user in lines[i]:
                modified_line = lines[i].replace(marker_user, "\n"+marker_user)
                result.append(modified_line)
            else:
                result.append(lines[i])
            i += 1
            
    if result:
        result[-1] = result[-1] + marker_assistant
        
    return "\n".join(result)