import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import re

class ModelConfigurator(tk.Tk):
    """
    GUI application to configure hyperparameters for transformer models.
    Allows selecting presets and saving/loading configurations.
    """
    
    def __init__(self):
        """Initialize the main window and widgets"""
        super().__init__()
        
        # Set up main window
        self.title("Model Configurator")  
        self.geometry("500x350")
        
        # Dictionaries to store settings
        self.config_settings = {
            "n_embd": 768,
            "vocab_size": 16384,  
            "max_length": 512,
            "n_head": 8,
            "n_layer": 8,
            "dropout": 0.0,
            
            # VRAM estimation settings
            "dataset_size_gb": 1.0, "batch_size": 32  # New fields for VRAM estimation

        }
        
        self.presets = {
            # Preset configurations  
            "GPT-2 Small": {"n_embd": 768, "vocab_size": 50257, "max_length": 1024, "n_head": 12, "n_layer": 12, "dropout": 0.1},
            "GPT-2 Medium": {"n_embd": 1024, "vocab_size": 50257, "max_length": 1024, "n_head": 16, "n_layer": 24, "dropout": 0.1},
            "GPT-2 Large": {"n_embd": 1280, "vocab_size": 50257, "max_length": 1024, "n_head": 20, "n_layer": 36, "dropout": 0.1},
            "GPT-2 XL": {"n_embd": 1600, "vocab_size": 50257, "max_length": 1024, "n_head": 25, "n_layer": 48, "dropout": 0.1},
            "GPT-3 Ada": {"n_embd": 2560, "vocab_size": 50257, "max_length": 2048, "n_head": 32, "n_layer": 48, "dropout": 0.1},
            "GPT-3 Babbage": {"n_embd": 4096, "vocab_size": 50257, "max_length": 2048, "n_head": 40, "n_layer": 64, "dropout": 0.1},
            "GPT-3 Curie": {"n_embd": 6144, "vocab_size": 50257, "max_length": 2048, "n_head": 48, "n_layer": 96, "dropout": 0.1},
            "GPT-3 Davinci": {"n_embd": 12288, "vocab_size": 50257, "max_length": 2048, "n_head": 96, "n_layer": 192, "dropout": 0.1},
            "BERT Base": {"n_embd": 768, "vocab_size": 30522, "max_length": 512, "n_head": 12, "n_layer": 12, "dropout": 0.1},
            "BERT Large": {"n_embd": 1024, "vocab_size": 30522, "max_length": 512, "n_head": 16, "n_layer": 24, "dropout": 0.1},
            "RoBERTa Base": {"n_embd": 768, "vocab_size": 50265, "max_length": 512, "n_head": 12, "n_layer": 12, "dropout": 0.1},
            "RoBERTa Large": {"n_embd": 1024, "vocab_size": 50265, "max_length": 512, "n_head": 16, "n_layer": 24, "dropout": 0.1},
            "T5 Small": {"n_embd": 512, "vocab_size": 32128, "max_length": 512, "n_head": 8, "n_layer": 6, "dropout": 0.1},
            "ELECTRA Small": {"n_embd": 256, "vocab_size": 30522, "max_length": 512, "n_head": 4, "n_layer": 12, "dropout": 0.1},
            "XLNet Base": {"n_embd": 768, "vocab_size": 32000, "max_length": 512, "n_head": 12, "n_layer": 12, "dropout": 0.1},
            "DistilBERT Base": {"n_embd": 768, "vocab_size": 30522, "max_length": 512, "n_head": 12, "n_layer": 6, "dropout": 0.1},
            "ALBERT Base": {"n_embd": 768, "vocab_size": 30000, "max_length": 512, "n_head": 12, "n_layer": 12, "dropout": 0.0},
            "Transformer-XL Base": {"n_embd": 1024, "vocab_size": 267735, "max_length": 512, "n_head": 16, "n_layer": 18, "dropout": 0.1},
            "BART Large": {"n_embd": 1024, "vocab_size": 50265, "max_length": 1024, "n_head": 16, "n_layer": 12, "dropout": 0.1},
            "GPT-Neo 2.7B": {"n_embd": 2048, "vocab_size": 50257, "max_length": 2048, "n_head": 16, "n_layer": 32, "dropout": 0.1},
            "GPT-J 6B": {"n_embd": 4096, "vocab_size": 50257, "max_length": 2048, "n_head": 16, "n_layer": 28, "dropout": 0.1},
            "ERNIE 2.0 Base": {"n_embd": 768, "vocab_size": 30522, "max_length": 512, "n_head": 12, "n_layer": 12, "dropout": 0.1},
            "DeBERTa Large": {"n_embd": 1024, "vocab_size": 30522, "max_length": 512, "n_head": 16, "n_layer": 24, "dropout": 0.1}
        }
        
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        """Create and layout all widgets"""
        
        # Start from row 1 for presets (row 0 is used for the label)
        row = 1
        
        # Preset selection
        ttk.Label(self, text="Select a Preset:").grid(row=0, column=0, sticky="w")  
        self.preset_combobox = ttk.Combobox(self, values=list(self.presets.keys()), state="readonly") 
        self.preset_combobox.grid(row=0, column=1, sticky="ew")
        
        # Bind preset change event
        self.preset_combobox.bind("<<ComboboxSelected>>", self.apply_preset)   
        
        # Config entries
        self.entries = {} 
        for idx, (setting, value) in enumerate(self.config_settings.items(), start=row):
            ttk.Label(self, text=setting).grid(row=idx, column=0, sticky="w")
            entry = ttk.Entry(self)
            entry.insert(0, str(value))
            entry.grid(row=idx, column=1)
            self.entries[setting] = entry
            row = idx + 1  # Increment row for the next widget
            
        # Adjust row for Save/Load buttons to be below the last config entry
        row += 1
    
        # Save/load buttons
        self.save_button = ttk.Button(self, text="Save Config", command=self.save_config)  
        self.save_button.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
    
        self.load_button = ttk.Button(self, text="Load Config", command=self.load_config)
        self.load_button.grid(row=row, column=0, columnspan=2)
        row += 1
    
        # VRAM Estimation button
        self.estimate_vram_button = ttk.Button(self, text="Estimate VRAM", command=self.estimate_vram)
        self.estimate_vram_button.grid(row=row, column=0, columnspan=2, pady=10)


    def estimate_vram(self):
        # Simplified VRAM estimation formula
        # Note: This formula is illustrative and should be adjusted based on empirical data and specific model architecture
        model_size = sum(int(self.entries[setting].get()) for setting in ["n_embd", "vocab_size", "n_layer"]) / 1e6  # Simplified model size in GB
        dataset_size = float(self.entries["dataset_size_gb"].get())
        batch_size = int(self.entries["batch_size"].get())
        vram_usage = (model_size * batch_size + dataset_size) * 1.2  # Assuming a 20% overhead

        messagebox.showinfo("VRAM Estimation", f"Estimated VRAM Required: {vram_usage:.2f} GB")


    def apply_preset(self, event=None):
        """When preset selected, apply config"""
        
        selected_preset = self.preset_combobox.get()
        if selected_preset in self.presets:
            for setting, value in self.presets[selected_preset].items():
                self.config_settings[setting] = value
                self.entries[setting].delete(0, tk.END)
                self.entries[setting].insert(0, str(value))

    def is_valid(self, setting, value):
        """Check if value is valid for setting"""
        
        validators = {
            "n_embd": lambda x: re.fullmatch(r"\d+", x),
            "vocab_size": lambda x: re.fullmatch(r"\d+", x), 
            "max_length": lambda x: re.fullmatch(r"\d+", x),
            "n_head": lambda x: re.fullmatch(r"\d+", x),
            "n_layer": lambda x: re.fullmatch(r"\d+", x),
            "dropout": lambda x: 0 <= float(x) <= 1
        }
        
        validator_func = validators.get(setting) 
        if validator_func:
            return validator_func(str(value))
        else:
            return True

    def update_setting(self, setting, value):
        """Validate and update setting"""
        
        if self.is_valid(setting, value):
            if setting in ["n_embd", "vocab_size", "max_length", "n_head", "n_layer"]:
                self.config_settings[setting] = int(value)
            elif setting == "dropout":
                self.config_settings[setting] = float(value)
        else:
            messagebox.showerror("Error", f"Invalid value for {setting}")
            self.entries[setting].delete(0, tk.END)
            self.entries[setting].insert(0, self.config_settings[setting])

    def save_config(self):
        """Save configuration to file"""
        
        if all(self.is_valid(s, e.get()) for s, e in self.entries.items()):
            file_path = filedialog.asksaveasfilename(defaultextension=".json")
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.config_settings, f, indent=4) 
        else:
            messagebox.showerror("Error", "Invalid settings")

    def load_config(self): 
        """Load configuration from file"""
        
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                loaded_settings = json.load(f)
            for setting, value in loaded_settings.items():
                if setting in self.entries:
                    self.update_setting(setting, str(value))  
        
if __name__ == "__main__":
    app = ModelConfigurator()
    app.mainloop()
