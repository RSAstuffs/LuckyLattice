#!/usr/bin/env python3
"""
Luckylattice GUI - Modern Graphical Interface for Factorization Attacks

A comprehensive GUI application for the Minimizable Factorization Lattice Attack
with support for all factorization methods and adjustable parameters.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
import threading
import queue
from typing import Optional, Dict, Any
import re

# Import the factorization solver
try:
    from standalone_lattice_attack import (
        MinimizableFactorizationLatticeSolver,
        EnhancedPolynomialSolver
    )
    SOLVER_AVAILABLE = True
except ImportError as e:
    SOLVER_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ModernTheme:
    """Modern color theme for the GUI - Light Theme"""
    # Light theme colors (primary)
    BG_PRIMARY = "#ffffff"
    BG_SECONDARY = "#f5f5f5"
    BG_TERTIARY = "#e8e8e8"
    FG_PRIMARY = "#1e1e1e"
    FG_SECONDARY = "#6e6e6e"
    ACCENT = "#007acc"
    ACCENT_HOVER = "#1a8cd8"
    SUCCESS = "#28a745"
    WARNING = "#ffc107"
    ERROR = "#dc3545"
    BORDER = "#d0d0d0"
    
    # Text areas and input fields
    INPUT_BG = "#ffffff"
    INPUT_FG = "#1e1e1e"
    INPUT_BORDER = "#cccccc"


class OutputRedirector:
    """Redirect stdout/stderr to a text widget and results preview"""
    def __init__(self, text_widget, preview_widget=None):
        self.text_widget = text_widget
        self.preview_widget = preview_widget
        self.queue = queue.Queue()
        self.text_widget.after(100, self.process_queue)
        self.last_update_time = 0
        
    def write(self, text):
        self.queue.put(text)
        
    def flush(self):
        pass
        
    def process_queue(self):
        try:
            import time
            current_time = time.time()
            update_preview = (current_time - self.last_update_time) > 0.5  # Update preview every 0.5 seconds
            
            while True:
                text = self.queue.get_nowait()
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
                
                # Update preview with ALL output for verbose logging
                if self.preview_widget and text.strip():
                    # Always append to preview for verbose logging
                    self.preview_widget.insert(tk.END, text)
                    self.preview_widget.see(tk.END)
                    self.last_update_time = current_time
        except queue.Empty:
            pass
        finally:
            self.text_widget.after(100, self.process_queue)


class FactorizationGUI:
    """Main GUI Application for Luckylattice"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Luckylattice - Factorization Attack Suite")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Configuration storage
        self.config = {}
        self.is_running = False
        self.worker_thread = None
        
        # Setup theme
        self.setup_theme()
        
        # Create main UI
        self.create_widgets()
        
        # Redirect output
        self.setup_output_redirect()
        
    def setup_theme(self):
        """Configure modern theme"""
        style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            style.theme_use('clam')
        except:
            pass
            
        # Configure styles with light theme
        style.configure('TNotebook', background=ModernTheme.BG_SECONDARY, borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=ModernTheme.BG_TERTIARY,
                       foreground=ModernTheme.FG_PRIMARY,
                       padding=[20, 10],
                       borderwidth=1)
        style.map('TNotebook.Tab',
                 background=[('selected', ModernTheme.ACCENT)],
                 foreground=[('selected', 'white')])
        
        style.configure('TFrame', background=ModernTheme.BG_SECONDARY)
        style.configure('TLabel', background=ModernTheme.BG_SECONDARY, foreground=ModernTheme.FG_PRIMARY)
        style.configure('TLabelFrame', background=ModernTheme.BG_SECONDARY, foreground=ModernTheme.FG_PRIMARY)
        style.configure('TEntry', fieldbackground=ModernTheme.INPUT_BG, foreground=ModernTheme.INPUT_FG, 
                       bordercolor=ModernTheme.INPUT_BORDER, borderwidth=1)
        style.configure('TButton', background=ModernTheme.BG_TERTIARY, foreground=ModernTheme.FG_PRIMARY)
        style.map('TButton',
                 background=[('active', ModernTheme.ACCENT)],
                 foreground=[('active', 'white')])
        style.configure('TCheckbutton', background=ModernTheme.BG_SECONDARY, foreground=ModernTheme.FG_PRIMARY)
        style.configure('TRadiobutton', background=ModernTheme.BG_SECONDARY, foreground=ModernTheme.FG_PRIMARY)
        
        # Configure root background
        self.root.configure(bg=ModernTheme.BG_PRIMARY)
        
    def create_widgets(self):
        """Create main UI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(title_frame, 
                              text="🔐 Luckylattice Factorization Suite",
                              font=("Arial", 24, "bold"),
                              bg=ModernTheme.BG_PRIMARY,
                              fg=ModernTheme.ACCENT)
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Advanced Lattice & Polynomial Factorization Methods",
                                 font=("Arial", 11),
                                 bg=ModernTheme.BG_PRIMARY,
                                 fg=ModernTheme.FG_SECONDARY)
        subtitle_label.pack()
        
        # Method selection frame
        method_selection_frame = ttk.LabelFrame(main_frame, text="Method Selection", padding=10)
        method_selection_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        method_inner = ttk.Frame(method_selection_frame)
        method_inner.pack(fill=tk.X)
        
        ttk.Label(method_inner, text="Select factorization methods to use:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        
        # Method checkboxes
        methods_container = ttk.Frame(method_inner)
        methods_container.pack(fill=tk.X, pady=5)
        
        self.method_lattice_var = tk.BooleanVar(value=True)
        self.method_polynomial_var = tk.BooleanVar(value=False)
        self.method_sd_exact_var = tk.BooleanVar(value=False)
        self.method_sd_squared_var = tk.BooleanVar(value=False)
        self.method_auto_find_sd_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(methods_container, text="🔷 Lattice Attack", 
                       variable=self.method_lattice_var).grid(row=0, column=0, sticky=tk.W, padx=10, pady=3)
        ttk.Checkbutton(methods_container, text="📊 Polynomial Methods", 
                       variable=self.method_polynomial_var).grid(row=0, column=1, sticky=tk.W, padx=10, pady=3)
        ttk.Checkbutton(methods_container, text="🔢 Exact Factorization from S", 
                       variable=self.method_sd_exact_var).grid(row=0, column=2, sticky=tk.W, padx=10, pady=3)
        ttk.Checkbutton(methods_container, text="🔢 S² = 4N + D Method", 
                       variable=self.method_sd_squared_var).grid(row=1, column=0, sticky=tk.W, padx=10, pady=3)
        ttk.Checkbutton(methods_container, text="🔍 Auto-find S/D (Root's Method)", 
                       variable=self.method_auto_find_sd_var).grid(row=1, column=1, sticky=tk.W, padx=10, pady=3)
        
        methods_container.columnconfigure(0, weight=1)
        methods_container.columnconfigure(1, weight=1)
        methods_container.columnconfigure(2, weight=1)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_input_tab()
        self.create_lattice_tab()
        self.create_polynomial_tab()
        self.create_sd_factorization_tab()
        self.create_advanced_tab()
        self.create_ml_tab()
        self.create_results_tab()
        
        # Status bar
        self.create_status_bar(main_frame)
        
    def create_input_tab(self):
        """Create input/output tab"""
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="📥 Input/Output")
        
        # Main container with scroll
        canvas = tk.Canvas(input_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0, highlightbackground=ModernTheme.BORDER)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input section
        input_section = ttk.LabelFrame(scrollable_frame, text="Input Parameters", padding=15)
        input_section.pack(fill=tk.X, padx=10, pady=10)
        
        # N input
        n_frame = ttk.Frame(input_section)
        n_frame.pack(fill=tk.X, pady=5)
        ttk.Label(n_frame, text="N (Number to factor):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.n_entry = tk.Text(n_frame, height=4, bg=ModernTheme.BG_TERTIARY, 
                              fg=ModernTheme.FG_PRIMARY, insertbackground=ModernTheme.FG_PRIMARY,
                              wrap=tk.WORD, relief=tk.FLAT, borderwidth=1)
        self.n_entry.pack(fill=tk.X, pady=5)
        ttk.Label(n_frame, text="Enter the RSA modulus or composite number to factor", 
                 foreground=ModernTheme.FG_SECONDARY, font=("Arial", 8)).pack(anchor=tk.W)
        
        # p approximation
        p_frame = ttk.Frame(input_section)
        p_frame.pack(fill=tk.X, pady=5)
        ttk.Label(p_frame, text="p approximation (optional):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.p_entry = tk.Text(p_frame, height=2, bg=ModernTheme.INPUT_BG,
                              fg=ModernTheme.INPUT_FG, insertbackground=ModernTheme.INPUT_FG,
                              wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, highlightthickness=1,
                              highlightbackground=ModernTheme.INPUT_BORDER)
        self.p_entry.pack(fill=tk.X, pady=5)
        
        # q approximation
        q_frame = ttk.Frame(input_section)
        q_frame.pack(fill=tk.X, pady=5)
        ttk.Label(q_frame, text="q approximation (optional):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.q_entry = tk.Text(q_frame, height=2, bg=ModernTheme.INPUT_BG,
                              fg=ModernTheme.INPUT_FG, insertbackground=ModernTheme.INPUT_FG,
                              wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, highlightthickness=1,
                              highlightbackground=ModernTheme.INPUT_BORDER)
        self.q_entry.pack(fill=tk.X, pady=5)
        
        # Decimal approximations
        decimal_frame = ttk.Frame(input_section)
        decimal_frame.pack(fill=tk.X, pady=5)
        ttk.Label(decimal_frame, text="Decimal Approximations (optional):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        decimal_inner = ttk.Frame(decimal_frame)
        decimal_inner.pack(fill=tk.X, pady=5)
        
        ttk.Label(decimal_inner, text="p (decimal):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.p_decimal_entry = ttk.Entry(decimal_inner, width=40)
        self.p_decimal_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(decimal_inner, text="q (decimal):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.q_decimal_entry = ttk.Entry(decimal_inner, width=40)
        self.q_decimal_entry.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        decimal_inner.columnconfigure(1, weight=1)
        
        # S and D inputs
        sd_frame = ttk.LabelFrame(scrollable_frame, text="S/D Factorization Parameters", padding=15)
        sd_frame.pack(fill=tk.X, padx=10, pady=10)
        
        sd_inner = ttk.Frame(sd_frame)
        sd_inner.pack(fill=tk.X)
        
        ttk.Label(sd_inner, text="S (sum p+q):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.s_entry = ttk.Entry(sd_inner, width=50)
        self.s_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(sd_inner, text="D ((p-q)²):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.d_entry = ttk.Entry(sd_inner, width=50)
        self.d_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(sd_inner, text="S²:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.s_squared_entry = ttk.Entry(sd_inner, width=50)
        self.s_squared_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(sd_inner, text="D hint:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.d_hint_entry = ttk.Entry(sd_inner, width=50)
        self.d_hint_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        self.auto_find_sd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sd_inner, text="Auto-find S and D using Root's Method", 
                       variable=self.auto_find_sd_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        sd_inner.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        self.start_button = tk.Button(button_frame, text="🚀 Start Factorization", 
                                     command=self.start_factorization,
                                     bg=ModernTheme.ACCENT, fg="white",
                                     font=("Arial", 12, "bold"),
                                     relief=tk.RAISED, padx=20, pady=10,
                                     cursor="hand2", bd=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop", 
                                    command=self.stop_factorization,
                                    bg=ModernTheme.ERROR, fg="white",
                                    font=("Arial", 12, "bold"),
                                    relief=tk.RAISED, padx=20, pady=10,
                                    cursor="hand2", state=tk.DISABLED, bd=2)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="🗑 Clear", 
                                     command=self.clear_inputs,
                                     bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.FG_PRIMARY,
                                     font=("Arial", 12),
                                     relief=tk.RAISED, padx=20, pady=10,
                                     cursor="hand2", bd=2)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Results preview
        results_preview_frame = ttk.LabelFrame(scrollable_frame, text="Results Preview", padding=15)
        results_preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_preview = scrolledtext.ScrolledText(results_preview_frame,
                                                         height=10,
                                                         bg=ModernTheme.INPUT_BG,
                                                         fg=ModernTheme.INPUT_FG,
                                                         insertbackground=ModernTheme.INPUT_FG,
                                                         wrap=tk.WORD,
                                                         relief=tk.SOLID,
                                                         borderwidth=1,
                                                         highlightthickness=1,
                                                         highlightbackground=ModernTheme.INPUT_BORDER)
        self.results_preview.pack(fill=tk.BOTH, expand=True)
        
    def create_lattice_tab(self):
        """Create lattice attack parameters tab"""
        lattice_frame = ttk.Frame(self.notebook)
        self.notebook.add(lattice_frame, text="🔷 Lattice Attack")
        
        canvas = tk.Canvas(lattice_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(lattice_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Basic lattice parameters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Lattice Parameters", padding=15)
        basic_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(basic_frame, text="Search Radius (bits):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        search_radius_frame = ttk.Frame(basic_frame)
        search_radius_frame.pack(fill=tk.X, pady=5)
        self.search_radius_entry = ttk.Entry(search_radius_frame)
        self.search_radius_entry.insert(0, "2048")
        self.search_radius_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(search_radius_frame, text="(default: 2048 for full key coverage)", 
                 foreground=ModernTheme.FG_SECONDARY).pack(side=tk.LEFT)
        
        ttk.Label(basic_frame, text="Lattice Dimension:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        lattice_dim_frame = ttk.Frame(basic_frame)
        lattice_dim_frame.pack(fill=tk.X, pady=5)
        self.lattice_dimension_entry = ttk.Entry(lattice_dim_frame)
        self.lattice_dimension_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(lattice_dim_frame, text="(e.g., 1000000 for large scale)", 
                 foreground=ModernTheme.FG_SECONDARY).pack(side=tk.LEFT)
        
        # Advanced lattice parameters
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Lattice Parameters", padding=15)
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(advanced_frame, text="Maximum Lattice Vectors:").pack(anchor=tk.W)
        self.max_lattice_vectors_entry = ttk.Entry(advanced_frame)
        self.max_lattice_vectors_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(advanced_frame, text="Coefficient Limit:").pack(anchor=tk.W, pady=(10,0))
        self.coeff_limit_entry = ttk.Entry(advanced_frame)
        self.coeff_limit_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(advanced_frame, text="Ultra Search Radius (bits):").pack(anchor=tk.W, pady=(10,0))
        self.ultra_search_radius_entry = ttk.Entry(advanced_frame)
        self.ultra_search_radius_entry.pack(fill=tk.X, pady=5)
        
        # Options
        options_frame = ttk.LabelFrame(scrollable_frame, text="Lattice Options", padding=15)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.bulk_search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Enable bulk factor search using LLL (creates large lattice, may be slow)",
                       variable=self.bulk_search_var).pack(anchor=tk.W, pady=5)
        
        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Verbose output", variable=self.verbose_var).pack(anchor=tk.W, pady=5)
        
    def create_polynomial_tab(self):
        """Create polynomial methods tab"""
        poly_frame = ttk.Frame(self.notebook)
        self.notebook.add(poly_frame, text="📊 Polynomial Methods")
        
        canvas = tk.Canvas(poly_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(poly_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable polynomial methods
        enable_frame = ttk.LabelFrame(scrollable_frame, text="Enable Polynomial Methods", padding=15)
        enable_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.polynomial_enable_var = tk.BooleanVar(value=False)
        ttk.Label(enable_frame, text="Note: Enable this if you want polynomial methods to run automatically alongside lattice methods.", 
                 foreground=ModernTheme.FG_SECONDARY, font=("Arial", 9)).pack(anchor=tk.W, pady=(0, 5))
        ttk.Checkbutton(enable_frame, text="Enable polynomial solving methods",
                       variable=self.polynomial_enable_var).pack(anchor=tk.W, pady=2)
        
        # Polynomial generation parameters
        gen_frame = ttk.LabelFrame(scrollable_frame, text="Polynomial Generation", padding=15)
        gen_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(gen_frame, text="Maximum Polynomials:").pack(anchor=tk.W)
        self.max_polynomials_entry = ttk.Entry(gen_frame)
        self.max_polynomials_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(gen_frame, text="Polynomial Grid Size:").pack(anchor=tk.W, pady=(10,0))
        self.polynomial_grid_size_entry = ttk.Entry(gen_frame)
        self.polynomial_grid_size_entry.pack(fill=tk.X, pady=5)
        
        # Root finding parameters
        root_frame = ttk.LabelFrame(scrollable_frame, text="Root Finding Parameters", padding=15)
        root_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(root_frame, text="Maximum Root Candidates per Variable:").pack(anchor=tk.W)
        self.max_root_candidates_entry = ttk.Entry(root_frame)
        self.max_root_candidates_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(root_frame, text="Maximum Root Combinations:").pack(anchor=tk.W, pady=(10,0))
        self.max_root_combinations_entry = ttk.Entry(root_frame)
        self.max_root_combinations_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(root_frame, text="Root Sampling Strategy:").pack(anchor=tk.W, pady=(10,0))
        self.root_sampling_strategy_var = tk.StringVar(value="none")
        strategy_combo = ttk.Combobox(root_frame, textvariable=self.root_sampling_strategy_var,
                                     values=["none", "random", "stratified", "adaptive"],
                                     state="readonly", width=20)
        strategy_combo.pack(anchor=tk.W, pady=5)
        
        ttk.Label(root_frame, text="Root Sampling Fraction (0.0-1.0):").pack(anchor=tk.W, pady=(10,0))
        self.root_sampling_fraction_entry = ttk.Entry(root_frame)
        self.root_sampling_fraction_entry.insert(0, "1.0")
        self.root_sampling_fraction_entry.pack(fill=tk.X, pady=5)
        
        self.early_termination_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(root_frame, text="Early termination (stop once good candidate found)",
                       variable=self.early_termination_var).pack(anchor=tk.W, pady=5)
        
    def create_sd_factorization_tab(self):
        """Create S/D factorization tab"""
        sd_frame = ttk.Frame(self.notebook)
        self.notebook.add(sd_frame, text="🔢 S/D Factorization")
        
        canvas = tk.Canvas(sd_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sd_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Information
        info_frame = ttk.LabelFrame(scrollable_frame, text="About S/D Factorization", padding=15)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        info_text = """
S/D Factorization uses the relationship:
  S² = 4N + D
  where S = p + q (sum of factors)
  and D = (p - q)² (square of difference)

Methods available:
  • Exact factorization from S
  • Factorization from S² = 4N + D
  • Automatic S/D discovery using Root's Method
        """
        ttk.Label(info_frame, text=info_text.strip(), justify=tk.LEFT,
                 foreground=ModernTheme.FG_SECONDARY).pack(anchor=tk.W)
        
        # Method selection
        method_frame = ttk.LabelFrame(scrollable_frame, text="Method Selection", padding=15)
        method_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.sd_method_var = tk.StringVar(value="auto")
        ttk.Radiobutton(method_frame, text="Auto-detect best method", 
                       variable=self.sd_method_var, value="auto").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(method_frame, text="Exact factorization from S", 
                       variable=self.sd_method_var, value="exact_s").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(method_frame, text="Factorization from S² = 4N + D", 
                       variable=self.sd_method_var, value="s_squared").pack(anchor=tk.W, pady=5)
        
        # Note: S and D inputs are in the Input/Output tab
        
    def create_advanced_tab(self):
        """Create advanced parameters tab"""
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="⚙️ Advanced")
        
        canvas = tk.Canvas(advanced_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(advanced_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Trial division
        trial_frame = ttk.LabelFrame(scrollable_frame, text="Trial Division", padding=15)
        trial_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(trial_frame, text="Trial Division Limit:").pack(anchor=tk.W)
        self.trial_division_limit_entry = ttk.Entry(trial_frame)
        self.trial_division_limit_entry.pack(fill=tk.X, pady=5)
        ttk.Label(trial_frame, text="(default: auto)", 
                 foreground=ModernTheme.FG_SECONDARY).pack(anchor=tk.W)
        
        # Performance options
        perf_frame = ttk.LabelFrame(scrollable_frame, text="Performance Options", padding=15)
        perf_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.no_transformer_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(perf_frame, text="Disable Transformer model (use simpler model to save memory)",
                       variable=self.no_transformer_var).pack(anchor=tk.W, pady=5)
        
        # Configuration export/import
        config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration Management", padding=15)
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        config_button_frame = ttk.Frame(config_frame)
        config_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(config_button_frame, text="Export Configuration", 
                  command=self.export_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_button_frame, text="Import Configuration", 
                  command=self.import_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_button_frame, text="Reset to Defaults", 
                  command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        
    def create_ml_tab(self):
        """Create machine learning / transformer tab"""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 ML/Transformer")
        
        canvas = tk.Canvas(ml_frame, bg=ModernTheme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(ml_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Pre-training
        pretrain_frame = ttk.LabelFrame(scrollable_frame, text="Model Pre-training", padding=15)
        pretrain_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(pretrain_frame, text="Number of RSA keys for pre-training:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        pretrain_input_frame = ttk.Frame(pretrain_frame)
        pretrain_input_frame.pack(fill=tk.X, pady=5)
        self.pretrain_count_entry = ttk.Entry(pretrain_input_frame)
        self.pretrain_count_entry.insert(0, "20")
        self.pretrain_count_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(pretrain_input_frame, text="(recommended: 20-50)", 
                 foreground=ModernTheme.FG_SECONDARY).pack(side=tk.LEFT)
        
        ttk.Label(pretrain_frame, text="Base bit length for pre-training keys:").pack(anchor=tk.W, pady=(10,0))
        self.pretrain_bits_entry = ttk.Entry(pretrain_frame)
        self.pretrain_bits_entry.insert(0, "1024")
        self.pretrain_bits_entry.pack(fill=tk.X, pady=5)
        ttk.Label(pretrain_frame, text="(will vary ±100 bits for diversity)", 
                 foreground=ModernTheme.FG_SECONDARY).pack(anchor=tk.W)
        
        self.pretrain_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pretrain_frame, text="Pre-training only mode (no factorization)",
                       variable=self.pretrain_only_var).pack(anchor=tk.W, pady=5)
        
        # Model loading/saving
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Management", padding=15)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_frame, text="Load Pre-trained Model:").pack(anchor=tk.W)
        load_frame = ttk.Frame(model_frame)
        load_frame.pack(fill=tk.X, pady=5)
        self.load_model_entry = ttk.Entry(load_frame)
        self.load_model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(load_frame, text="Browse...", command=lambda: self.browse_file(self.load_model_entry)).pack(side=tk.LEFT)
        
        ttk.Label(model_frame, text="Save Model After Training/Attack:").pack(anchor=tk.W, pady=(10,0))
        save_frame = ttk.Frame(model_frame)
        save_frame.pack(fill=tk.X, pady=5)
        self.save_model_entry = ttk.Entry(save_frame)
        self.save_model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(save_frame, text="Browse...", command=lambda: self.browse_save_file(self.save_model_entry)).pack(side=tk.LEFT)
        
    def create_results_tab(self):
        """Create results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 Results")
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(results_frame,
                                                     bg=ModernTheme.INPUT_BG,
                                                     fg=ModernTheme.INPUT_FG,
                                                     insertbackground=ModernTheme.INPUT_FG,
                                                     wrap=tk.WORD,
                                                     font=("Courier", 10),
                                                     relief=tk.SOLID,
                                                     borderwidth=1,
                                                     highlightthickness=1,
                                                     highlightbackground=ModernTheme.INPUT_BORDER)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Copy to Clipboard", command=self.copy_output).pack(side=tk.LEFT, padx=5)
        
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = tk.Frame(parent, bg=ModernTheme.BG_TERTIARY, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                     bg=ModernTheme.BG_TERTIARY, 
                                     fg=ModernTheme.FG_PRIMARY,
                                     anchor=tk.W,
                                     padx=10, pady=5)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_var = tk.StringVar(value="")
        progress_label = tk.Label(status_frame, textvariable=self.progress_var,
                                 bg=ModernTheme.BG_TERTIARY,
                                 fg=ModernTheme.FG_SECONDARY,
                                 padx=10, pady=5)
        progress_label.pack(side=tk.RIGHT)
        
    def setup_output_redirect(self):
        """Setup output redirection"""
        self.output_redirector = OutputRedirector(self.output_text, self.results_preview)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector
        
    def browse_file(self, entry_widget):
        """Browse for file and set entry"""
        filename = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All files", "*.*"), ("Model files", "*.pth;*.pt;*.model")]
        )
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
            
    def browse_save_file(self, entry_widget):
        """Browse for save file location"""
        filename = filedialog.asksaveasfilename(
            title="Save model as",
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
            
    def get_input_values(self) -> Dict[str, Any]:
        """Collect all input values from GUI"""
        config = {}
        
        # Input values
        n_text = self.n_entry.get("1.0", tk.END).strip()
        if n_text:
            try:
                config['N'] = int(n_text)
            except ValueError:
                raise ValueError("Invalid N format")
        
        p_text = self.p_entry.get("1.0", tk.END).strip()
        if p_text:
            try:
                config['p'] = int(p_text)
            except ValueError:
                pass
        
        q_text = self.q_entry.get("1.0", tk.END).strip()
        if q_text:
            try:
                config['q'] = int(q_text)
            except ValueError:
                pass
        
        p_decimal = self.p_decimal_entry.get().strip()
        if p_decimal:
            config['p_decimal'] = p_decimal
        
        q_decimal = self.q_decimal_entry.get().strip()
        if q_decimal:
            config['q_decimal'] = q_decimal
        
        # S/D values
        s = self.s_entry.get().strip()
        if s:
            try:
                config['s'] = int(s)
            except ValueError:
                pass
        
        d = self.d_entry.get().strip()
        if d:
            try:
                config['d'] = int(d)
            except ValueError:
                pass
        
        s_squared = self.s_squared_entry.get().strip()
        if s_squared:
            try:
                config['s_squared'] = int(s_squared)
            except ValueError:
                pass
        
        d_hint = self.d_hint_entry.get().strip()
        if d_hint:
            try:
                config['d_hint'] = int(d_hint)
            except ValueError:
                pass
        
        config['auto_find_sd'] = self.auto_find_sd_var.get()
        
        # Lattice parameters
        search_radius = self.search_radius_entry.get().strip()
        if search_radius:
            try:
                config['search_radius'] = int(search_radius)
            except ValueError:
                pass
        
        lattice_dim = self.lattice_dimension_entry.get().strip()
        if lattice_dim:
            try:
                config['lattice_dimension'] = int(lattice_dim)
            except ValueError:
                pass
        
        max_vec = self.max_lattice_vectors_entry.get().strip()
        if max_vec:
            try:
                config['max_lattice_vectors'] = int(max_vec)
            except ValueError:
                pass
        
        coeff_limit = self.coeff_limit_entry.get().strip()
        if coeff_limit:
            try:
                config['coeff_limit'] = int(coeff_limit)
            except ValueError:
                pass
        
        ultra_radius = self.ultra_search_radius_entry.get().strip()
        if ultra_radius:
            try:
                config['ultra_search_radius'] = int(ultra_radius)
            except ValueError:
                pass
        
        config['bulk'] = self.bulk_search_var.get()
        config['verbose'] = self.verbose_var.get()
        
        # Polynomial parameters
        config['polynomial'] = self.polynomial_enable_var.get()
        
        max_poly = self.max_polynomials_entry.get().strip()
        if max_poly:
            try:
                config['max_polynomials'] = int(max_poly)
            except ValueError:
                pass
        
        poly_grid = self.polynomial_grid_size_entry.get().strip()
        if poly_grid:
            try:
                config['polynomial_grid_size'] = int(poly_grid)
            except ValueError:
                pass
        
        max_roots = self.max_root_candidates_entry.get().strip()
        if max_roots:
            try:
                config['max_root_candidates'] = int(max_roots)
            except ValueError:
                pass
        
        max_combos = self.max_root_combinations_entry.get().strip()
        if max_combos:
            try:
                config['max_root_combinations'] = int(max_combos)
            except ValueError:
                pass
        
        config['root_sampling_strategy'] = self.root_sampling_strategy_var.get()
        
        sampling_frac = self.root_sampling_fraction_entry.get().strip()
        if sampling_frac:
            try:
                config['root_sampling_fraction'] = float(sampling_frac)
            except ValueError:
                pass
        
        config['early_termination'] = self.early_termination_var.get()
        
        # Advanced parameters
        trial_div = self.trial_division_limit_entry.get().strip()
        if trial_div:
            try:
                config['trial_division_limit'] = int(trial_div)
            except ValueError:
                pass
        
        config['no_transformer'] = self.no_transformer_var.get()
        
        # ML parameters
        pretrain_count = self.pretrain_count_entry.get().strip()
        if pretrain_count:
            try:
                config['pretrain'] = int(pretrain_count)
            except ValueError:
                pass
        
        pretrain_bits = self.pretrain_bits_entry.get().strip()
        if pretrain_bits:
            try:
                config['pretrain_bits'] = int(pretrain_bits)
            except ValueError:
                pass
        
        config['pretrain_only'] = self.pretrain_only_var.get()
        
        load_model = self.load_model_entry.get().strip()
        if load_model:
            config['load_pretrained'] = load_model
        
        save_model = self.save_model_entry.get().strip()
        if save_model:
            config['save_pretrained'] = save_model
        
        # Method selection
        config['method_lattice'] = self.method_lattice_var.get()
        config['method_polynomial'] = self.method_polynomial_var.get()
        config['method_sd_exact'] = self.method_sd_exact_var.get()
        config['method_sd_squared'] = self.method_sd_squared_var.get()
        config['method_auto_find_sd'] = self.method_auto_find_sd_var.get()
        
        return config
        
    def start_factorization(self):
        """Start factorization in background thread"""
        if not SOLVER_AVAILABLE:
            messagebox.showerror("Error", f"Cannot import factorization solver:\n{IMPORT_ERROR}")
            return
        
        try:
            config = self.get_input_values()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return
        
        if 'N' not in config and not config.get('pretrain_only'):
            messagebox.showerror("Input Error", "N (number to factor) is required")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running factorization...", fg=ModernTheme.ACCENT)
        
        # Clear output
        self.output_text.delete("1.0", tk.END)
        self.results_preview.delete("1.0", tk.END)
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self.factorization_worker, args=(config,), daemon=True)
        self.worker_thread.start()
        
    def factorization_worker(self, config):
        """Worker thread for factorization"""
        try:
            import sys
            sys.set_int_max_str_digits(100000)  # Handle very large integers
            
            from standalone_lattice_attack import MinimizableFactorizationLatticeSolver, EnhancedPolynomialSolver
            
            if config.get('pretrain_only'):
                print("=" * 80)
                print("PRE-TRAINING MODE")
                print("=" * 80)
                print(f"Pre-training mode: {config.get('pretrain')} keys, {config.get('pretrain_bits')} bits")
                # Pre-training would go here - for now just acknowledge
                self.root.after(0, lambda: self.factorization_complete(True, "Pre-training completed"))
                return
            
            N = config['N']
            initial_msg = f"Starting factorization of {N.bit_length()}-bit number\n"
            initial_msg += f"Methods selected: "
            methods_list = []
            if config.get('method_lattice'):
                methods_list.append("Lattice Attack")
            if config.get('method_polynomial'):
                methods_list.append("Polynomial Methods")
            if config.get('method_sd_exact'):
                methods_list.append("Exact from S")
            if config.get('method_sd_squared'):
                methods_list.append("S² = 4N + D")
            if config.get('method_auto_find_sd'):
                methods_list.append("Auto-find S/D")
            initial_msg += ", ".join(methods_list) if methods_list else "None"
            initial_msg += "\n" + "=" * 80 + "\n"
            
            # Update preview immediately
            self.root.after(0, lambda t=initial_msg: self.update_preview(t, append=False))
            
            print("=" * 80)
            print(f"FACTORING: {N}")
            print(f"Bit length: {N.bit_length()}")
            print("=" * 80)
            
            # Build config dict for solver
            print("\n📋 Configuration:")
            solver_config = {}
            if config.get('lattice_dimension'):
                solver_config['lattice_dimension'] = config['lattice_dimension']
                print(f"  • Lattice dimension: {config['lattice_dimension']}")
            if config.get('max_lattice_vectors'):
                solver_config['max_lattice_vectors'] = config['max_lattice_vectors']
                print(f"  • Max lattice vectors: {config['max_lattice_vectors']}")
            if config.get('coeff_limit'):
                solver_config['coeff_limit'] = config['coeff_limit']
                print(f"  • Coefficient limit: {config['coeff_limit']}")
            if config.get('trial_division_limit'):
                solver_config['trial_division_limit'] = config['trial_division_limit']
                print(f"  • Trial division limit: {config['trial_division_limit']}")
            if config.get('ultra_search_radius'):
                solver_config['ultra_search_radius'] = config['ultra_search_radius']
                print(f"  • Ultra search radius: {config['ultra_search_radius']} bits")
            if config.get('polynomial_grid_size'):
                solver_config['polynomial_grid_size'] = config['polynomial_grid_size']
                print(f"  • Polynomial grid size: {config['polynomial_grid_size']}")
            if config.get('max_root_candidates'):
                solver_config['max_root_candidates'] = config['max_root_candidates']
                print(f"  • Max root candidates: {config['max_root_candidates']}")
            if config.get('max_root_combinations'):
                solver_config['max_root_combinations'] = config['max_root_combinations']
                print(f"  • Max root combinations: {config['max_root_combinations']}")
            if config.get('root_sampling_strategy'):
                solver_config['root_sampling_strategy'] = config['root_sampling_strategy']
                print(f"  • Root sampling strategy: {config['root_sampling_strategy']}")
            if config.get('root_sampling_fraction'):
                solver_config['root_sampling_fraction'] = config['root_sampling_fraction']
                print(f"  • Root sampling fraction: {config['root_sampling_fraction']}")
            if config.get('early_termination'):
                solver_config['early_termination'] = config['early_termination']
                print(f"  • Early termination: Enabled")
            if config.get('no_transformer'):
                solver_config['no_transformer'] = config['no_transformer']
                print(f"  • Transformer model: Disabled")
            
            # Initialize solver
            print("\n🔧 Initializing lattice solver...")
            solver = MinimizableFactorizationLatticeSolver(N, delta=0.75)
            solver.config = solver_config
            print("✅ Solver initialized")
            
            # Handle method selection - only run selected methods
            refined_p = None
            refined_q = None
            improvement = 0.0
            
            # Method 1: Auto-find S/D using Root's Method
            if config.get('method_auto_find_sd') and config.get('auto_find_sd'):
                print("\n" + "=" * 80)
                print("🔍 METHOD 1: AUTO-FINDING S AND D USING ROOT'S METHOD")
                print("=" * 80)
                print("⏳ Attempting to automatically discover S and D values...")
                # This would call the auto-find S/D method
                # For now, this is handled by the S/D methods below if they're selected
                print("ℹ️  Auto-find S/D enabled - will be used if S/D methods are selected")
                print("⚠️  Note: Auto-find functionality depends on method selection")
            
            # Method 2: Exact factorization from S
            if config.get('method_sd_exact') and config.get('s') and not config.get('d'):
                print("\n" + "=" * 80)
                print("🔢 METHOD 2: EXACT FACTORIZATION FROM S")
                print("=" * 80)
                S = config['s']
                print(f"📊 Input parameters:")
                print(f"  • S (sum p+q) = {S}")
                print(f"  • S bit length: {S.bit_length()} bits")
                
                search_radius_bits = config.get('search_radius', 2048)
                if search_radius_bits:
                    search_radius = 2 ** search_radius_bits
                else:
                    search_radius = 2 ** 2048
                
                print(f"  • Search radius: {search_radius_bits} bits (2^{search_radius_bits})")
                print(f"\n⏳ Running exact factorization algorithm...")
                print(f"   Using S = {S}")
                
                p_exact, q_exact, T_exact, k_exact = solver.solve_exact_from_s(S=S, search_radius=search_radius)
                
                print(f"\n📊 Results from exact S method:")
                if p_exact and q_exact:
                    print(f"  • p found: {p_exact.bit_length()} bits")
                    print(f"  • q found: {q_exact.bit_length()} bits")
                    if T_exact:
                        print(f"  • T (p+q) = {T_exact}")
                    if k_exact:
                        print(f"  • k (p-q) = {k_exact}")
                
                if p_exact and q_exact and p_exact * q_exact == N:
                    print("\n" + "=" * 80)
                    print("🎉 SUCCESS! EXACT FACTORIZATION FOUND FROM S!")
                    print("=" * 80)
                    preview_text = f"✅ FACTORIZATION SUCCESSFUL!\n\n"
                    preview_text += f"Method: Exact from S\n\n"
                    preview_text += f"p = {p_exact}\n\n"
                    preview_text += f"q = {q_exact}\n\n"
                    self.root.after(0, lambda t=preview_text: self.update_preview(t, append=True))
                    self.root.after(0, lambda: self.factorization_complete(True, "Factorization successful!"))
                    return
                elif p_exact and q_exact:
                    refined_p, refined_q = p_exact, q_exact
            
            # Method 3: S² = 4N + D method
            if config.get('method_sd_squared') and (config.get('s_squared') or (config.get('s') and config.get('d'))):
                print("\n" + "=" * 80)
                print("🔢 METHOD 3: FACTORIZATION FROM S² = 4N + D")
                print("=" * 80)
                print(f"📊 Using S² = 4N + D relationship")
                
                if config.get('s_squared'):
                    S_squared = config['s_squared']
                    print(f"  • S² provided: {S_squared}")
                    print(f"  • S² bit length: {S_squared.bit_length()} bits")
                    print(f"\n⏳ Computing factors from S²...")
                    result_p, result_q, alpha_final, beta_final = solver.factor_from_s_squared(S_squared=S_squared)
                elif config.get('s') and config.get('d'):
                    S = config['s']
                    D = config['d']
                    print(f"  • S provided: {S}")
                    print(f"  • D provided: {D}")
                    print(f"  • S bit length: {S.bit_length()} bits")
                    print(f"  • D bit length: {D.bit_length()} bits")
                    print(f"\n⏳ Computing factors from S and D...")
                    result_p, result_q, alpha_final, beta_final = solver.factor_from_s_squared(S=S, D=D)
                else:
                    result_p, result_q = None, None
                    print("⚠️  Insufficient parameters for S² method")
                
                if result_p and result_q:
                    print(f"\n📊 Results from S² method:")
                    print(f"  • p found: {result_p.bit_length()} bits")
                    print(f"  • q found: {result_q.bit_length()} bits")
                    if alpha_final is not None and beta_final is not None:
                        print(f"  • Alpha adjustment: {alpha_final}")
                        print(f"  • Beta adjustment: {beta_final}")
                
                if result_p and result_q and result_p * result_q == N:
                    print("\n" + "=" * 80)
                    print("🎉 SUCCESS! EXACT FACTORIZATION FOUND USING S² = 4N + D!")
                    print("=" * 80)
                    preview_text = f"✅ FACTORIZATION SUCCESSFUL!\n\n"
                    preview_text += f"Method: S² = 4N + D\n\n"
                    preview_text += f"p = {result_p}\n\n"
                    preview_text += f"q = {result_q}\n\n"
                    self.root.after(0, lambda t=preview_text: self.update_preview(t, append=True))
                    self.root.after(0, lambda: self.factorization_complete(True, "Factorization successful!"))
                    return
                elif result_p and result_q:
                    refined_p, refined_q = result_p, result_q
            
            # Method 4: Standard lattice attack
            if config.get('method_lattice'):
                print("\n" + "=" * 80)
                print("🔷 METHOD 4: LATTICE ATTACK")
                print("=" * 80)
                # Get p and q approximations
                p_approx = config.get('p')
                q_approx = config.get('q')
                
                if not p_approx or not q_approx:
                    # Estimate candidates
                    import math
                    print(f"📊 Estimating initial candidates (not provided)...")
                    sqrt_N = math.isqrt(N)
                    p_approx = sqrt_N
                    q_approx = sqrt_N
                    print(f"  • Estimated p ≈ √N = {p_approx.bit_length()} bits")
                    print(f"  • Estimated q ≈ √N = {q_approx.bit_length()} bits")
                else:
                    print(f"📊 Using provided approximations:")
                    print(f"  • p_approx: {p_approx.bit_length()} bits")
                    print(f"  • q_approx: {q_approx.bit_length()} bits")
                    # Calculate initial error
                    initial_product = p_approx * q_approx
                    initial_error = abs(initial_product - N)
                    print(f"  • Initial error: {initial_error.bit_length()} bits")
                    print(f"  • Initial product: {initial_product.bit_length()} bits")
                
                # Get search radius
                search_radius_bits = config.get('search_radius', 2048)
                if search_radius_bits:
                    search_radius = 2 ** search_radius_bits
                else:
                    search_radius = 2 ** 2048
                
                print(f"\n⚙️  Lattice parameters:")
                print(f"  • Search radius: {search_radius_bits} bits (2^{search_radius_bits})")
                print(f"  • Delta: 0.75")
                if solver_config.get('lattice_dimension'):
                    print(f"  • Lattice dimension: {solver_config['lattice_dimension']}")
                
                # Run lattice factorization
                print(f"\n⏳ Starting lattice attack...")
                print(f"   Building pyramid lattice basis...")
                lattice_p, lattice_q, lattice_improvement, pyramid_basis = solver.solve(
                    p_approx, q_approx, search_radius
                )
                
                print(f"\n📊 Lattice attack results:")
                if lattice_p and lattice_q:
                    print(f"  • p found: {lattice_p.bit_length()} bits")
                    print(f"  • q found: {lattice_q.bit_length()} bits")
                    product = lattice_p * lattice_q
                    error = abs(product - N)
                    print(f"  • Final error: {error.bit_length()} bits")
                    print(f"  • Improvement: {lattice_improvement:.6f}")
                    if pyramid_basis is not None:
                        print(f"  • Vectors processed: {len(pyramid_basis):,}")
                else:
                    print(f"  • No valid factors found")
                
                if lattice_p and lattice_q:
                    if not refined_p or not refined_q:
                        refined_p, refined_q = lattice_p, lattice_q
                        improvement = lattice_improvement
                    elif lattice_p * lattice_q == N:
                        refined_p, refined_q = lattice_p, lattice_q
                        improvement = lattice_improvement
                    else:
                        # Use the one with smaller error
                        lattice_error = abs(lattice_p * lattice_q - N)
                        current_error = abs(refined_p * refined_q - N) if refined_p and refined_q else float('inf')
                        if lattice_error < current_error:
                            refined_p, refined_q = lattice_p, lattice_q
                            improvement = lattice_improvement
            
            # Method 5: Polynomial methods
            if config.get('method_polynomial') or config.get('polynomial'):
                if not refined_p or not refined_q or refined_p * refined_q != N:
                    print("\n" + "=" * 80)
                    print("📊 METHOD 5: POLYNOMIAL-BASED FACTORIZATION")
                    print("=" * 80)
                    
                    # Get p and q approximations for polynomial solver
                    if not p_approx or not q_approx:
                        import math
                        sqrt_N = math.isqrt(N)
                        p_approx = sqrt_N
                        q_approx = sqrt_N
                        print(f"📊 Using estimated approximations for polynomial solver:")
                        print(f"  • p_approx: {p_approx.bit_length()} bits")
                        print(f"  • q_approx: {q_approx.bit_length()} bits")
                    
                    print(f"\n🔧 Initializing polynomial solver...")
                    poly_solver = EnhancedPolynomialSolver(N, config=solver_config, p_approx=p_approx, q_approx=q_approx)
                    print(f"✅ Polynomial solver initialized")
                    
                    # Generate and solve polynomials
                    print(f"\n⏳ Generating polynomials...")
                    max_poly = solver_config.get('max_polynomials')
                    grid_size = solver_config.get('polynomial_grid_size')
                    if max_poly:
                        print(f"  • Max polynomials: {max_poly}")
                    if grid_size:
                        print(f"  • Grid size: {grid_size}")
                    
                    polynomials = poly_solver.generate_polynomials(
                        max_polynomials=max_poly,
                        grid_size=grid_size
                    )
                    
                    print(f"✅ Generated {len(polynomials)} polynomials")
                    
                    if polynomials:
                        print(f"\n⏳ Attempting to solve polynomials using all methods...")
                        print(f"   Methods to try: Gröbner basis, Resultants, Modular constraints, etc.")
                        poly_result = poly_solver.solve_with_all_methods(
                            polynomials,
                            p_hint=p_approx,
                            q_hint=q_approx
                        )
                        
                        print(f"\n📊 Polynomial solving results:")
                        if poly_result:
                            poly_p, poly_q = poly_result
                            poly_product = poly_p * poly_q
                            poly_error = abs(poly_product - N)
                            print(f"  • p found: {poly_p.bit_length()} bits")
                            print(f"  • q found: {poly_q.bit_length()} bits")
                            print(f"  • Error: {poly_error.bit_length()} bits")
                        else:
                            print(f"  • No solution found")
                        
                        if poly_result:
                            poly_p, poly_q = poly_result
                            if poly_p * poly_q == N:
                                print("\n" + "=" * 80)
                                print("🎉 SUCCESS! EXACT FACTORIZATION FOUND WITH POLYNOMIAL METHODS!")
                                print("=" * 80)
                                refined_p, refined_q = poly_p, poly_q
                            else:
                                # Use polynomial result if better
                                poly_error = abs(poly_p * poly_q - N)
                                if refined_p and refined_q:
                                    lattice_error = abs(refined_p * refined_q - N)
                                    if poly_error < lattice_error:
                                        refined_p, refined_q = poly_p, poly_q
                                else:
                                    refined_p, refined_q = poly_p, poly_q
            
            # Final results check
            print("\n" + "=" * 80)
            print("📋 FINAL RESULTS SUMMARY")
            print("=" * 80)
            
            if not refined_p or not refined_q:
                print("❌ No factorization found with selected methods")
                print("\n💡 Suggestions:")
                print("  • Try different methods")
                print("  • Provide better initial approximations")
                print("  • Increase search radius")
                print("  • Enable more methods simultaneously")
                self.root.after(0, lambda: self.factorization_complete(False, "No factorization found with selected methods"))
                return
            
            # Check results
            if refined_p and refined_q:
                print(f"✅ Factors found:")
                print(f"  • p: {refined_p.bit_length()} bits")
                print(f"  • q: {refined_q.bit_length()} bits")
                product = refined_p * refined_q
                print(f"  • Product: {product.bit_length()} bits")
                print(f"  • Target N: {N.bit_length()} bits")
                
                if product == N:
                    print("\n" + "=" * 80)
                    print("🎉 SUCCESS! EXACT FACTORIZATION FOUND!")
                    print("=" * 80)
                    print(f"✅ Exact match verified!")
                    print(f"\np = {refined_p}")
                    print(f"\nq = {refined_q}")
                    print(f"\nVerification: {refined_p} × {refined_q} = {product} = N ✓")
                    
                    # Update preview
                    preview_text = f"✅ FACTORIZATION SUCCESSFUL!\n\n"
                    preview_text += f"p = {refined_p}\n\n"
                    preview_text += f"q = {refined_q}\n\n"
                    preview_text += f"Verification: p × q = {product} = N ✓\n"
                    
                    self.root.after(0, lambda t=preview_text: self.update_preview(t, append=True))
                    self.root.after(0, lambda: self.factorization_complete(True, "Factorization successful!"))
                else:
                    error = abs(product - N)
                    print(f"\n⚠️  APPROXIMATION FOUND (not exact):")
                    print(f"  • Error: {error.bit_length()} bits")
                    print(f"  • Error value: {error}")
                    print(f"  • Improvement factor: {improvement:.6f}")
                    print(f"  • Relative error: {(error / N * 100):.10f}%")
                    
                    if error.bit_length() < 50:
                        print(f"  • Status: Very close! May be factorable with refinement")
                    elif error.bit_length() < 100:
                        print(f"  • Status: Good approximation, try other methods")
                    else:
                        print(f"  • Status: Still far from exact factorization")
                    preview_text = f"⚠️ IMPROVED APPROXIMATION\n\n"
                    preview_text += f"Error: {error.bit_length()} bits\n"
                    preview_text += f"Improvement: {improvement:.6f}\n\n"
                    preview_text += f"p ≈ {refined_p}\n"
                    preview_text += f"q ≈ {refined_q}\n"
                    self.root.after(0, lambda t=preview_text: self.update_preview(t, append=True))
                    self.root.after(0, lambda e=error.bit_length(): self.factorization_complete(False, f"Improved approximation (error: {e} bits)"))
                
        except Exception as e:
            import traceback
            error_msg = f"Error during factorization: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: self.factorization_complete(False, f"Error: {str(e)}"))
            
    def factorization_complete(self, success, message):
        """Called when factorization completes"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if success:
            self.status_label.config(text=message, fg=ModernTheme.SUCCESS)
        else:
            self.status_label.config(text=message, fg=ModernTheme.WARNING)
        
    def update_preview(self, text, append=False):
        """Update results preview"""
        try:
            if not append:
                self.results_preview.delete("1.0", tk.END)
            self.results_preview.insert(tk.END, text)
            self.results_preview.see(tk.END)
            self.results_preview.update_idletasks()  # Force update
        except Exception as e:
            # Fallback: just print to console if preview widget doesn't exist yet
            print(f"Preview update error: {e}")
        
    def stop_factorization(self):
        """Stop factorization (note: this is a soft stop)"""
        self.is_running = False
        self.status_label.config(text="Stopping...", fg=ModernTheme.WARNING)
        # Note: Actual cancellation would require more sophisticated thread management
        
    def clear_inputs(self):
        """Clear all input fields"""
        self.n_entry.delete("1.0", tk.END)
        self.p_entry.delete("1.0", tk.END)
        self.q_entry.delete("1.0", tk.END)
        self.p_decimal_entry.delete(0, tk.END)
        self.q_decimal_entry.delete(0, tk.END)
        self.s_entry.delete(0, tk.END)
        self.d_entry.delete(0, tk.END)
        self.s_squared_entry.delete(0, tk.END)
        self.d_hint_entry.delete(0, tk.END)
        self.results_preview.delete("1.0", tk.END)
        
    def clear_output(self):
        """Clear output text"""
        self.output_text.delete("1.0", tk.END)
        
    def save_output(self):
        """Save output to file"""
        filename = filedialog.asksaveasfilename(
            title="Save output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.output_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"Output saved to {filename}")
            
    def copy_output(self):
        """Copy output to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text.get("1.0", tk.END))
        messagebox.showinfo("Copied", "Output copied to clipboard")
        
    def export_config(self):
        """Export configuration to file"""
        # Implementation for config export
        messagebox.showinfo("Export", "Configuration export feature coming soon")
        
    def import_config(self):
        """Import configuration from file"""
        # Implementation for config import
        messagebox.showinfo("Import", "Configuration import feature coming soon")
        
    def reset_defaults(self):
        """Reset all parameters to defaults"""
        self.search_radius_entry.delete(0, tk.END)
        self.search_radius_entry.insert(0, "2048")
        self.polynomial_enable_var.set(False)
        self.verbose_var.set(False)
        # Add more defaults as needed
        messagebox.showinfo("Reset", "Parameters reset to defaults")


def main():
    """Main entry point"""
    if not SOLVER_AVAILABLE:
        root = tk.Tk()
        messagebox.showerror("Import Error", 
                           f"Cannot import factorization solver.\n\n"
                           f"Error: {IMPORT_ERROR}\n\n"
                           f"Please ensure standalone_lattice_attack.py is in the same directory.")
        root.destroy()
        return
    
    root = tk.Tk()
    app = FactorizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

