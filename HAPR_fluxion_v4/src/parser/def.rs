//! DEF (Design Exchange Format) writer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use log::info;

use crate::data::circuit::Circuit;
use crate::data::placement::Placement;

/// Write placement result to a basic DEF file.
pub fn write_def(path: &Path, circuit: &Circuit, placement: &Placement) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create DEF: {}", e))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "VERSION 5.8 ;").unwrap();
    writeln!(writer, "DIVIDERCHAR \"/\" ;").unwrap();
    writeln!(writer, "BUSBITCHARS \"[]\" ;").unwrap();
    writeln!(writer, "DESIGN {} ;", circuit.name).unwrap();
    writeln!(writer, "UNITS DISTANCE MICRONS 1000 ;").unwrap();
    writeln!(writer, "\nDIEAREA ( 0 0 ) ( {} {} ) ;", 
        (circuit.chip_width * 1000.0) as i64, 
        (circuit.chip_height * 1000.0) as i64
    ).unwrap();

    writeln!(writer, "\nCOMPONENTS {} ;", circuit.num_gates).unwrap();
    for i in 0..circuit.num_gates {
        let name = &circuit.gate_names[i];
        let x = (placement.gate_x[i] * 1000.0) as i64;
        let y = (placement.gate_y[i] * 1000.0) as i64;
        
        // Output format: - name type + PLACED ( x y ) N ;
        writeln!(writer, "    - {} GATE_TYPE + PLACED ( {} {} ) N ;", name, x, y).unwrap();
    }
    writeln!(writer, "END COMPONENTS ;").unwrap();

    writeln!(writer, "\nEND DESIGN").unwrap();
    
    info!("Wrote DEF to {}", path.display());
    Ok(())
}
