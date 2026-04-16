import matplotlib.pyplot as plt


def generate_isolines(lon, lat, val, levels):
    contours = plt.contour(lon, lat, val, levels=levels)
    isolines = []
    for paths in contours.get_paths():
        isolines.append(paths.vertices.tolist())
    return isolines


def format_isoline_raw(isolines, levels):
    out = {}
    for i, level in enumerate(levels):
        out[level] = isolines[i]
    return out


def format_isoline_geojson(isolines, levels):
    features = []
    for i, level in enumerate(levels):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(coord[0]), float(coord[1])] for coord in isolines[i]
                    ],
                },
                "properties": {"id": i, "level": level},
            }
        )
    return {"type": "FeatureCollection", "features": features}
