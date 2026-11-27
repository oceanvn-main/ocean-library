#!/usr/bin/env python3
"""
transform_editor_2d.py

Interactive 2D Transform Editor.
Szeliski Computer Vision Exercise 2.2.
- Open an empty window (canvas).
- Shift-drag (rubber-band) to create a new rectangle.
- Select deformation mode (motion model): translation, rigid, similarity, affine, or perspective.
- Drag any corner of the outline to change its transformation.
- Supports persistence (save and load transformations as JSON).

Author: Antigravity AI
Date: 2026-06-10
"""

import sys
import os
import json
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np

# Dark Theme Palette
COLOR_BG = "#111113"
COLOR_PANEL = "#17191d"
COLOR_CANVAS = "#101216"
COLOR_GRID = "#25272d"
COLOR_TEXT = "#e0e0e0"
COLOR_ACCENT = "#3498db"
COLOR_SECONDARY = "#e74c3c"
COLOR_MUTED = "#55585d"
COLOR_ACTIVE_SHAPE = "#00bcff"
COLOR_ACTIVE_FILL = "#15334d"
COLOR_INACTIVE_SHAPE = "#7f8c8d"
COLOR_INACTIVE_FILL = "#2c3e50"


class Rectangle2D:
    def __init__(self, base_corners):
        # base_corners: 4x2 numpy array representing original rectangle coordinates
        self.base_corners = np.array(base_corners, dtype=float)
        # H: 3x3 homography matrix (initially identity)
        self.H = np.eye(3, dtype=float)

    def get_current_corners(self):
        # Project base corners using H
        pts_hom = np.hstack([self.base_corners, np.ones((4, 1))])  # 4x3
        warped = (self.H @ pts_hom.T).T  # 4x3
        # Divide by homogeneous coordinate w
        w = warped[:, 2:3]
        # Avoid division by zero
        w = np.where(np.abs(w) < 1e-15, 1e-15 * np.sign(w), w)
        return warped[:, :2] / w

    def fit_from_corners(self, mode, dst_corners):
        # dst_corners: 4x2 target corners
        # Updates self.H by solving the selected transformation mapping self.base_corners -> dst_corners
        src = self.base_corners
        dst = dst_corners

        if mode == "Translation":
            # translation has 2 DOF (tx, ty)
            # Match center of gravity translation
            tx = np.mean(dst[:, 0]) - np.mean(src[:, 0])
            ty = np.mean(dst[:, 1]) - np.mean(src[:, 1])
            H = np.eye(3)
            H[0, 2] = tx
            H[1, 2] = ty
            self.H = H

        elif mode == "Rigid":
            # rigid has 3 DOF (theta, tx, ty)
            # Solve using Procrustes analysis
            c_src = np.mean(src, axis=0)
            c_dst = np.mean(dst, axis=0)
            src_c = src - c_src
            dst_c = dst - c_dst

            # Covariance matrix
            M = src_c.T @ dst_c
            U, _, Vt = np.linalg.svd(M)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T

            t = c_dst - R @ c_src
            H = np.eye(3)
            H[:2, :2] = R
            H[:2, 2] = t
            self.H = H

        elif mode == "Similarity":
            # similarity has 4 DOF (s*cos, s*sin, tx, ty)
            # Solve via linear least squares:
            # [ x  -y   1   0 ] [  a ]   [ u ]
            # [ y   x   0   1 ] [  b ] = [ v ]
            #                   [ tx ]
            #                   [ ty ]
            A = []
            b = []
            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A.append([x, -y, 1, 0])
                A.append([y,  x, 0, 1])
                b.extend([u, v])
            A = np.array(A)
            b = np.array(b)
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            H = np.eye(3)
            H[0, 0] = c[0]
            H[0, 1] = -c[1]
            H[1, 0] = c[1]
            H[1, 1] = c[0]
            H[0, 2] = c[2]
            H[1, 2] = c[3]
            self.H = H

        elif mode == "Affine":
            # affine has 6 DOF (a, b, tx, c, d, ty)
            # Solve using linear least squares:
            # [ x  y  1 ] [ a ] = [ u ]
            #             [ b ]
            #             [ tx]
            A = []
            b_u = []
            b_v = []
            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A.append([x, y, 1])
                b_u.append(u)
                b_v.append(v)
            A = np.array(A)
            c_u, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
            c_v, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)
            H = np.eye(3)
            H[0, :] = c_u
            H[1, :] = c_v
            self.H = H

        elif mode == "Perspective":
            # homography has 8 DOF
            # Solve using standard Direct Linear Transform (DLT) with h22 = 1
            A = []
            b = []
            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
                A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
                b.extend([u, v])
            A = np.array(A)
            b = np.array(b)
            h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            H = np.ones(9)
            H[:8] = h
            self.H = H.reshape((3, 3))


class TransformEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2D Transform Editor — Szeliski Ex 2.2")
        self.geometry("1100x700")
        self.configure(bg=COLOR_BG)

        # State Variables
        self.rectangles = []
        self.selected_rect_idx = -1
        self.dragged_corner_idx = -1
        self.motion_mode = tk.StringVar(value="Translation")

        # Rubber-band state
        self.draw_start_x = None
        self.draw_start_y = None
        self.draw_temp_id = None
        self.is_shift_pressed = False

        self.setup_ui()
        self.draw_grid()

    def setup_ui(self):
        # Configure Grid Layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 1. Sidebar Control Panel
        sidebar = tk.Frame(self, bg=COLOR_PANEL, width=280)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        sidebar.grid_propagate(False)

        # Sidebar title
        tk.Label(sidebar, text="Motion Model", font=("Helvetica", 12, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=15, pady=(15, 5))
        
        # Mode Selection
        modes = ["Translation", "Rigid", "Similarity", "Affine", "Perspective"]
        for mode in modes:
            tk.Radiobutton(
                sidebar, text=mode, variable=self.motion_mode, value=mode,
                bg=COLOR_PANEL, fg=COLOR_TEXT, selectcolor=COLOR_PANEL,
                activebackground=COLOR_PANEL, activeforeground=COLOR_ACCENT,
                font=("Helvetica", 10), command=self.update_matrix_display
            ).pack(anchor="w", padx=25, pady=3)

        tk.Frame(sidebar, height=1, bg=COLOR_GRID).pack(fill="x", padx=15, pady=15)

        # Matrix Readout
        tk.Label(sidebar, text="Pose Matrix (H)", font=("Helvetica", 12, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=15, pady=(0, 5))
        
        self.matrix_frame = tk.Frame(sidebar, bg=COLOR_PANEL)
        self.matrix_frame.pack(fill="x", padx=20, pady=5)
        self.matrix_labels = []
        for r in range(3):
            row_labels = []
            for c in range(3):
                lbl = tk.Label(self.matrix_frame, text="0.00", font=("Courier", 10), bg=COLOR_CANVAS, fg=COLOR_TEXT, width=8, height=1, relief="ridge", borderwidth=1)
                lbl.grid(row=r, column=c, padx=3, pady=3)
                row_labels.append(lbl)
            self.matrix_labels.append(row_labels)

        tk.Frame(sidebar, height=1, bg=COLOR_GRID).pack(fill="x", padx=15, pady=15)

        # Persistence Buttons
        tk.Label(sidebar, text="Persistence", font=("Helvetica", 11, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=15, pady=(0, 5))
        btn_frame = tk.Frame(sidebar, bg=COLOR_PANEL)
        btn_frame.pack(fill="x", padx=15, pady=5)

        tk.Button(btn_frame, text="Save Pose", command=self.save_representation, bg=COLOR_ACCENT, fg="white", font=("Helvetica", 10, "bold"), relief="flat", activebackground="#2980b9").pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(btn_frame, text="Load Pose", command=self.load_representation, bg=COLOR_MUTED, fg="white", font=("Helvetica", 10, "bold"), relief="flat", activebackground="#7f8c8d").pack(side="left", fill="x", expand=True, padx=5)

        # Instructions label
        instr = (
            "Instructions:\n"
            "- Hold SHIFT & Drag to create a new rectangle.\n"
            "- Drag a corner handle of the selected rectangle to warp it.\n"
            "- Press Delete to remove the selected rectangle."
        )
        tk.Label(sidebar, text=instr, font=("Helvetica", 9), bg=COLOR_PANEL, fg=COLOR_MUTED, justify="left", wraplength=240).pack(side="bottom", anchor="w", padx=15, pady=15)

        # 2. Main Drawing Canvas
        self.canvas = tk.Canvas(self, bg=COLOR_CANVAS, highlightthickness=1, highlightbackground=COLOR_GRID)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Event Bindings
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<KeyPress-Shift_L>", self.set_shift_on)
        self.bind("<KeyPress-Shift_R>", self.set_shift_on)
        self.bind("<KeyRelease-Shift_L>", self.set_shift_off)
        self.bind("<KeyRelease-Shift_R>", self.set_shift_off)
        self.bind("<Delete>", self.delete_selected)
        self.bind("<BackSpace>", self.delete_selected)

    def draw_grid(self):
        self.canvas.update()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        grid_size = 50

        # Vertical grid lines
        for x in range(0, w, grid_size):
            self.canvas.create_line(x, 0, x, h, fill=COLOR_GRID, tags="grid")
        # Horizontal grid lines
        for y in range(0, h, grid_size):
            self.canvas.create_line(0, y, w, y, fill=COLOR_GRID, tags="grid")

    def set_shift_on(self, event):
        self.is_shift_pressed = True

    def set_shift_off(self, event):
        self.is_shift_pressed = False

    def on_mouse_down(self, event):
        x, y = event.x, event.y

        if self.is_shift_pressed:
            # Shift dragging starts rubber-band creation
            self.draw_start_x = x
            self.draw_start_y = y
            self.draw_temp_id = self.canvas.create_rectangle(x, y, x, y, outline=COLOR_ACTIVE_SHAPE, dash=(4, 4), width=1)
            return

        # Check if clicked close to a corner handle of the selected rectangle
        if self.selected_rect_idx != -1:
            rect = self.rectangles[self.selected_rect_idx]
            corners = rect.get_current_corners()
            for idx, pt in enumerate(corners):
                if np.linalg.norm(pt - [x, y]) < 10:  # 10px click radius
                    self.dragged_corner_idx = idx
                    return

        # Check if clicked inside any rectangle to select it
        for idx, rect in enumerate(reversed(self.rectangles)):
            # Reversed search so top shape is selected first
            corners = rect.get_current_corners()
            # Simple ray-casting/polygon inclusion logic
            if self.point_in_poly(x, y, corners):
                self.selected_rect_idx = len(self.rectangles) - 1 - idx
                self.update_canvas()
                self.update_matrix_display()
                return

        # Clicked on empty space -> deselect
        self.selected_rect_idx = -1
        self.update_canvas()
        self.update_matrix_display()

    def on_mouse_drag(self, event):
        x, y = event.x, event.y

        if self.is_shift_pressed and self.draw_temp_id is not None:
            # Update rubber band rectangle
            self.canvas.coords(self.draw_temp_id, self.draw_start_x, self.draw_start_y, x, y)
            return

        # Dragging corner of the selected shape
        if self.selected_rect_idx != -1 and self.dragged_corner_idx != -1:
            rect = self.rectangles[self.selected_rect_idx]
            mode = self.motion_mode.get()
            corners = rect.get_current_corners()
            k = self.dragged_corner_idx

            # Calculate the target corners Q based on the drag mode
            Q = corners.copy()

            if mode == "Translation":
                # Translate all corners
                dx = x - corners[k, 0]
                dy = y - corners[k, 1]
                Q += [dx, dy]

            elif mode == "Rigid":
                # Rotate other corners around the opposite corner
                opp = (k + 2) % 4
                p_opp = corners[opp]
                v_old = corners[k] - p_opp
                v_new = np.array([x, y]) - p_opp
                
                # Check for zero vector
                if np.linalg.norm(v_old) > 1e-5 and np.linalg.norm(v_new) > 1e-5:
                    theta = np.arctan2(v_new[1], v_new[0]) - np.arctan2(v_old[1], v_old[0])
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    for j in range(4):
                        if j != opp:
                            Q[j] = p_opp + R @ (corners[j] - p_opp)

            elif mode == "Similarity":
                # Scale & rotate around opposite corner (which stays fixed)
                opp = (k + 2) % 4
                p_opp = corners[opp]
                
                # Setup 4x4 similarity system mapping p_opp -> p_opp, and corners[k] -> mouse
                A = [
                    [p_opp[0], -p_opp[1], 1, 0],
                    [p_opp[1],  p_opp[0], 0, 1],
                    [corners[k, 0], -corners[k, 1], 1, 0],
                    [corners[k, 1],  corners[k, 0], 0, 1]
                ]
                b = [p_opp[0], p_opp[1], x, y]
                try:
                    c = np.linalg.solve(A, b)
                    S = np.array([[c[0], -c[1], c[2]], [c[1], c[0], c[3]], [0, 0, 1]])
                    for j in range(4):
                        p_hom = np.array([corners[j, 0], corners[j, 1], 1.0])
                        Q[j] = (S @ p_hom)[:2]
                except np.linalg.LinAlgError:
                    pass

            elif mode == "Affine":
                # Keep opposite and one adjacent corner fixed, map active corner to mouse
                opp = (k + 2) % 4
                adj = (k + 3) % 4
                p_opp = corners[opp]
                p_adj = corners[adj]
                
                # Solve 6-parameter affine system mapping p_opp->p_opp, p_adj->p_adj, corners[k]->mouse
                M = np.array([
                    [p_opp[0], p_opp[1], 1],
                    [p_adj[0], p_adj[1], 1],
                    [corners[k, 0], corners[k, 1], 1]
                ])
                try:
                    c_u = np.linalg.solve(M, [p_opp[0], p_adj[0], x])
                    c_v = np.linalg.solve(M, [p_opp[1], p_adj[1], y])
                    A = np.array([c_u, c_v, [0, 0, 1]])
                    for j in range(4):
                        p_hom = np.array([corners[j, 0], corners[j, 1], 1.0])
                        Q[j] = (A @ p_hom)[:2]
                except np.linalg.LinAlgError:
                    pass

            elif mode == "Perspective":
                # Pin all other 3 corners, warp the selected corner to mouse
                Q[k] = [x, y]

            # Fit self.H mapping original base_corners to the new corners Q
            rect.fit_from_corners(mode, Q)
            self.update_canvas()
            self.update_matrix_display()

    def on_mouse_up(self, event):
        x, y = event.x, event.y

        if self.is_shift_pressed and self.draw_temp_id is not None:
            # Delete temporary rubber band line
            self.canvas.delete(self.draw_temp_id)
            self.draw_temp_id = None

            # Calculate bounds
            x0, y0 = self.draw_start_x, self.draw_start_y
            w = abs(x - x0)
            h = abs(y - y0)
            
            # Avoid tiny clicks creating rectangles
            if w > 15 and h > 15:
                # Center point
                cx = (x + x0) / 2
                cy = (y + y0) / 2
                # Create base corners relative to center
                base_corners = np.array([
                    [-w/2, -h/2],
                    [ w/2, -h/2],
                    [ w/2,  h/2],
                    [-w/2,  h/2]
                ])
                # Instantiate rectangle
                rect = Rectangle2D(base_corners)
                # Translate it to center
                rect.H[0, 2] = cx
                rect.H[1, 2] = cy
                
                self.rectangles.append(rect)
                self.selected_rect_idx = len(self.rectangles) - 1
                self.update_canvas()
                self.update_matrix_display()
            return

        self.dragged_corner_idx = -1

    def delete_selected(self, event):
        if self.selected_rect_idx != -1:
            del self.rectangles[self.selected_rect_idx]
            self.selected_rect_idx = -1
            self.update_canvas()
            self.update_matrix_display()

    def update_canvas(self):
        # Clear previous rectangles and handles
        self.canvas.delete("shape")
        self.canvas.delete("handle")

        # Redraw all shapes
        for idx, rect in enumerate(self.rectangles):
            corners = rect.get_current_corners()
            pts_flat = corners.flatten().tolist()
            
            is_active = (idx == self.selected_rect_idx)
            color_shape = COLOR_ACTIVE_SHAPE if is_active else COLOR_INACTIVE_SHAPE
            color_fill = COLOR_ACTIVE_FILL if is_active else COLOR_INACTIVE_FILL
            width = 2.0 if is_active else 1.0

            # Draw polygon shape
            self.canvas.create_polygon(
                pts_flat, outline=color_shape, fill=color_fill, 
                width=width, tags="shape"
            )

            # Draw handles for the selected shape
            if is_active:
                for corner_idx, pt in enumerate(corners):
                    cx, cy = pt
                    r = 5
                    self.canvas.create_oval(
                        cx-r, cy-r, cx+r, cy+r, fill=COLOR_ACTIVE_SHAPE, 
                        outline="white", width=1.0, tags="handle"
                    )

    def update_matrix_display(self):
        if self.selected_rect_idx != -1:
            H = self.rectangles[self.selected_rect_idx].H
        else:
            H = np.eye(3)

        for r in range(3):
            for c in range(3):
                val = H[r, c]
                self.matrix_labels[r][c].configure(text=f"{val:.2f}")

    def point_in_poly(self, x, y, poly):
        # Ray casting algorithm to check point inside polygon
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def save_representation(self):
        # Save rectangles base coordinates and transforms to JSON
        data = []
        for rect in self.rectangles:
            data.append({
                "base_corners": rect.base_corners.tolist(),
                "H": rect.H.tolist()
            })
        
        filePath = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Results"),
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if filePath:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filePath), exist_ok=True)
                with open(filePath, "w") as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Success", f"Transform pose saved successfully to:\n{filePath}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save representation:\n{str(e)}")

    def load_representation(self):
        # Load rectangles and transforms from JSON
        filePath = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "Results"),
            filetypes=[("JSON Files", "*.json")]
        )
        if filePath:
            try:
                with open(filePath, "r") as f:
                    data = json.load(f)
                
                self.rectangles = []
                for item in data:
                    rect = Rectangle2D(item["base_corners"])
                    rect.H = np.array(item["H"])
                    self.rectangles.append(rect)
                
                self.selected_rect_idx = -1
                self.update_canvas()
                self.update_matrix_display()
                messagebox.showinfo("Success", f"Loaded {len(self.rectangles)} shapes successfully from:\n{filePath}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load representation:\n{str(e)}")


if __name__ == "__main__":
    app = TransformEditorApp()
    app.mainloop()
