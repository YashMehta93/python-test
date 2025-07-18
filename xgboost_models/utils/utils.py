from collections import defaultdict

def categorize_features(selected_features, full_features_map):
    """
    Categorize selected features into their respective groups from the full features map.
    """
    categorized_features = defaultdict(list)
    for feature in selected_features:
        for category, features in full_features_map.items():
            if feature in features:
                categorized_features[category].append(feature)
    return categorized_features

def flatten_categorized_features(categorized_features):
    """
    Flatten categorized features back into a flat list of selected features.
    """
    selected_features = []
    for category, features in categorized_features.items():
        selected_features.extend(features)
    return selected_features