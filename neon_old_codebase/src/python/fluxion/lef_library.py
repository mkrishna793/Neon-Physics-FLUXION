"""
FLUXION LEF Library Definitions

Provides physical dimension approximations for standard cells
across different technology nodes (3nm, 7nm, 14nm, 28nm).

In a full production flow, this would parse a real .lef file from a foundry.
For FLUXION, we use accurate dimension estimates to power the legalizer.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MacroDef:
    """Definition of a standard cell macro."""
    name: str
    width_um: float
    height_um: float
    pin_count: int


class LEFLibrary:
    """
    Standard cell library definition.
    """

    def __init__(self, node: str = "7nm", dbu_per_micron: int = 1000):
        """
        Initialize LEF library for a specific technology node.
        
        Args:
            node: Tech node string ("3nm", "7nm", "14nm", "28nm")
            dbu_per_micron: Database units per micron
        """
        self.node = node
        self.dbu = dbu_per_micron
        self.macros: Dict[str, MacroDef] = {}
        
        # Load parameters based on node
        if node == "3nm":
            self.row_height = 0.270
            self.site_width = 0.054
        elif node == "7nm":
            self.row_height = 0.360
            self.site_width = 0.054
        elif node == "14nm":
            self.row_height = 0.576
            self.site_width = 0.064
        else: # 28nm roughly
            self.row_height = 1.8
            self.site_width = 0.2
            
        self._build_library()

    def _build_library(self) -> None:
        """Build approximations for common standard cells."""
        # Multipliers based on standard cell complexity
        # X1 drive strength logic
        cell_types = {
            "INV":   (1, 2),  # 1 poly pitch
            "BUF":   (2, 2),  
            "NAND":  (2, 3),  
            "NOR":   (2, 3),
            "AND":   (3, 3),
            "OR":    (3, 3),
            "XOR":   (4, 3),
            "MUX":   (5, 4),
            "DFF":   (12, 4), # Larger sequential
            "ADD":   (18, 5), # Full adder
        }
        
        for base_type, (sites, pins) in cell_types.items():
            name = f"{base_type}_X1"
            width = sites * self.site_width
            height = self.row_height # 1 row high (except dual height cells)
            
            # Make DFFs dual-height if node is small
            if base_type == "DFF" and self.row_height < 0.5:
                height = self.row_height * 2
                width = (sites // 2) * self.site_width
                
            self.macros[base_type] = MacroDef(
                name=name, width_um=width, height_um=height, pin_count=pins
            )

    def get_macro_name(self, raw_type: str) -> str:
        """Map abstract logic type to concrete macro name."""
        # Simple string matching to find the base type
        raw_upper = raw_type.upper()
        
        for k in self.macros.keys():
            if k in raw_upper:
                return self.macros[k].name
                
        # Default fallback
        return f"{raw_upper}_X1"
        
    def get_macro_dimensions(self, raw_type: str) -> tuple:
        """Returns (width_um, height_um)."""
        raw_upper = raw_type.upper()
        
        for k, v in self.macros.items():
            if k in raw_upper:
                return v.width_um, v.height_um
                
        # Estimate if unknown
        return self.site_width * 4, self.row_height
