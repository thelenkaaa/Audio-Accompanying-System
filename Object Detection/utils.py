def filter_relevant_timings(object_timings, relevant_tags):
    """
    Filters the object timings dictionary to keep only relevant objects.

    :param object_timings: dict - Dictionary with object names as keys and lists of (start, end) tuples.
    :param relevant_tags: list - List of objects that are relevant for audio.
    :return: dict - Filtered dictionary containing only relevant objects.
    """
    return {obj: intervals for obj, intervals in object_timings.items() if obj in relevant_tags}


def group_object_detections(detections, min_gap=2.0, min_duration=1.0):
    """
    Groups object detections based on their screen times.
    If the gap between consecutive appearances of the same object exceeds `min_gap`,
    they are treated as separate occurrences.

    Args:
        detections (list of tuples): List of (object_id, start_time, end_time) detections.
        min_gap (float): Maximum allowed gap (in seconds) to merge detections.
        min_duration (float): Minimum duration required to consider an object for audio generation.
    
    Returns:
        dict: Grouped detections with merged time ranges per object.
    """
    object_timings = defaultdict(list)
    
    # Sort detections by object_id and start_time
    detections.sort(key=lambda x: (x[0], x[1]))
    
    for obj_id, start, end in detections:
        if object_timings[obj_id] and start - object_timings[obj_id][-1][1] <= min_gap:
            # Merge with the last time range if the gap is small
            object_timings[obj_id][-1] = (object_timings[obj_id][-1][0], max(end, object_timings[obj_id][-1][1]))
        else:
            # Start a new time range for this object
            object_timings[obj_id].append((start, end))
    
    # Filter out short durations and round times
    for obj_id in list(object_timings.keys()):
        object_timings[obj_id] = [(round(s, 2), round(e, 2)) for s, e in object_timings[obj_id] if e - s >= min_duration]
        if not object_timings[obj_id]:
            del object_timings[obj_id]  # Remove if no valid durations left
    
    return object_timings


def calculate_durations(object_timings):
    """
    Calculates the duration for each detected object based on its start and end times.

    :param object_timings: dict - Dictionary with object names as keys and lists of (start, end) tuples.
    :return: dict - Dictionary with object names as keys and their total screen duration in seconds.
    """
    object_durations = {}
    
    for obj, intervals in object_timings.items():
        total_duration = sum(end - start for start, end in intervals)
        object_durations[obj] = round(total_duration, 2)  # Round to 2 decimal places
    
    return object_durations