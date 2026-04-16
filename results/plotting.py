from pathlib import Path
from xml.sax.saxutils import escape


def _extent(values, fallback=(0.0, 1.0)):
    values = list(values)
    if not values:
        return fallback
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = 1.0 if minimum == 0 else abs(minimum) * 0.1
        return minimum - padding, maximum + padding
    return minimum, maximum


def _line_path(points):
    return " ".join(f"L {x:.2f} {y:.2f}" if i else f"M {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))


def save_line_plot(path, series, title, xlabel, ylabel, horizontal_lines=None):
    width, height = 800, 420
    left, right, top, bottom = 70, 30, 45, 60
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [x for values, _, _ in series for x in values]
    y_values = [y for _, values, _ in series for y in values]
    if horizontal_lines:
        y_values.extend(value for value, _ in horizontal_lines)

    x_min, x_max = _extent(x_values)
    y_min, y_max = _extent(y_values)

    def scale_x(value):
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def scale_y(value):
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-size="20" font-family="Arial">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" />',
        f'<text x="{width / 2:.0f}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="Arial">{escape(xlabel)}</text>',
        f'<text x="18" y="{height / 2:.0f}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 18 {height / 2:.0f})">{escape(ylabel)}</text>',
    ]

    for tick in range(5):
        x_value = x_min + (x_max - x_min) * tick / 4
        y_value = y_min + (y_max - y_min) * tick / 4
        x = scale_x(x_value)
        y = scale_y(y_value)
        svg.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="black" />')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{x_value:.3g}</text>')
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black" />')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{y_value:.3g}</text>')

    for value, label in horizontal_lines or []:
        y = scale_y(value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#888" stroke-dasharray="6 4" />')
        svg.append(f'<text x="{left + plot_width - 4}" y="{y - 6:.2f}" text-anchor="end" font-size="12" font-family="Arial">{escape(label)}</text>')

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    legend_y = top + 10

    for index, (x_series, y_series, label) in enumerate(series):
        points = [(scale_x(x), scale_y(y)) for x, y in zip(x_series, y_series)]
        color = colors[index % len(colors)]
        svg.append(f'<path d="{_line_path(points)}" fill="none" stroke="{color}" stroke-width="2.5" />')
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />')
        legend_x = left + index * 150
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2.5" />')
        svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-size="12" font-family="Arial">{escape(label)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_grouped_bar_chart(path, labels, series, title, xlabel, ylabel):
    width, height = 800, 420
    left, right, top, bottom = 70, 30, 45, 60
    plot_width = width - left - right
    plot_height = height - top - bottom

    y_values = [value for values, _ in series for value in values]
    y_min, y_max = 0.0, _extent(y_values, fallback=(0.0, 1.0))[1]

    def scale_y(value):
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-size="20" font-family="Arial">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" />',
        f'<text x="{width / 2:.0f}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="Arial">{escape(xlabel)}</text>',
        f'<text x="18" y="{height / 2:.0f}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 18 {height / 2:.0f})">{escape(ylabel)}</text>',
    ]

    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4
        y = scale_y(y_value)
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black" />')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{y_value:.3g}</text>')

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    cluster_width = plot_width / max(len(labels), 1)
    bar_width = cluster_width / (len(series) + 1)

    for label_index, label in enumerate(labels):
        cluster_x = left + label_index * cluster_width
        svg.append(
            f'<text x="{cluster_x + cluster_width / 2:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{escape(label)}</text>'
        )
        for series_index, (values, series_label) in enumerate(series):
            value = values[label_index]
            color = colors[series_index % len(colors)]
            x = cluster_x + series_index * bar_width + bar_width * 0.5
            y = scale_y(value)
            height_px = top + plot_height - y
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.8:.2f}" height="{height_px:.2f}" fill="{color}" />')

    legend_y = top + 10
    for index, (_, label) in enumerate(series):
        legend_x = left + index * 150
        color = colors[index % len(colors)]
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="24" height="12" fill="{color}" />')
        svg.append(f'<text x="{legend_x + 30}" y="{legend_y}" font-size="12" font-family="Arial">{escape(label)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")
