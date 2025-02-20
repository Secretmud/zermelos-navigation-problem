from manim import *
import numpy as np

def format_memory_size(bytes_used):
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = bytes_used
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f"{size:.2f} {units[unit]}"

def matrix_memory_usage(k):
    elements = 3**(2*k)
    bytes_used = elements * np.dtype(np.float16).itemsize
    return elements, format_memory_size(bytes_used)

class QubitMatrixGrowth(Scene):
    def construct(self):
        N = 20
        title = Text("Growth of Matrix Size & Memory Usage for Qubits").scale(0.75)
        title.to_edge(UR)
        self.play(Write(title))
        
        axes = Axes(
            x_range=[0, 25, 1],
            y_range=[0, 25, 1],
            axis_config={"color": WHITE},
        ).scale(0.75)
        labels = axes.get_axis_labels(x_label="Qubits", y_label="Memory Usage")
        
        self.play(Create(axes), Write(labels))
        
        points = []
        for k in range(2, N):
            elements, mem_size = matrix_memory_usage(k)
            y_val = np.log10(elements)  # Normalize for scaling
            dot = Dot(axes.c2p(k, y_val-2), color=BLUE)
            text = Text(mem_size, font_size=14).next_to(dot, UP, buff=0.1)
            
            self.play(FadeIn(dot), Write(text), run_time=0.5)
            points.append(dot)
        
        curve = VMobject(color=BLUE).set_points_smoothly([axes.c2p(i, np.log10(matrix_memory_usage(i-2)[0])) for i in range(N)])
        self.play(Create(curve))
        
        self.wait(2)