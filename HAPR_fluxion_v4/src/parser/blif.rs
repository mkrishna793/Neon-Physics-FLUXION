//! Robust BLIF (Berkeley Logic Interchange Format) Parser.
//!
//! Parses gate declarations (.gate or .names) and net connectivity.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use log::{info, warn};

use crate::data::circuit::{Circuit, GateType};

/// Parse a BLIF file and construct a Circuit.
pub fn parse(path: &Path) -> Result<Circuit, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open BLIF: {}", e))?;
    let reader = BufReader::new(file);

    let mut circuit = Circuit::new(
        path.file_stem().unwrap_or_default().to_string_lossy().as_ref(),
        1000.0, // Default chip width
        1000.0, // Default chip height
        10.0,   // Default row height
    );

    let mut gate_count = 0;
    // Map gate output net name -> driving gate index
    let mut net_to_driver = std::collections::HashMap::new();
    // List of (gate_index, input_net_names)
    let mut gate_inputs = Vec::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| e.to_string())?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens[0] == ".names" || tokens[0] == ".latch" || tokens[0] == ".gate" {
            let gate_type = if tokens[0] == ".latch" {
                GateType::Sequential
            } else {
                GateType::Combinational
            };

            let gate_name = format!("g{}", gate_count);
            // Assign some default size for logic gates
            let g_idx = circuit.add_gate(&gate_name, gate_type, 2.0, 10.0);
            gate_count += 1;

            if tokens[0] == ".names" {
                if tokens.len() >= 2 {
                    let out_net = tokens.last().unwrap().to_string();
                    net_to_driver.insert(out_net, g_idx);

                    let inputs: Vec<String> = tokens[1..tokens.len() - 1]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    gate_inputs.push((g_idx, inputs));
                }
            } else if tokens[0] == ".latch" {
                if tokens.len() >= 3 {
                    let in_net = tokens[1].to_string();
                    let out_net = tokens[2].to_string();
                    net_to_driver.insert(out_net, g_idx);
                    gate_inputs.push((g_idx, vec![in_net]));
                }
            } else if tokens[0] == ".gate" {
                // Example: .gate AND2 a=in1 b=in2 O=out
                let mut inputs = Vec::new();
                for token in &tokens[2..] {
                    if let Some((_, net_name)) = token.split_once('=') {
                        if token.starts_with("O=") || token.starts_with("Y=") {
                            net_to_driver.insert(net_name.to_string(), g_idx);
                        } else {
                            inputs.push(net_name.to_string());
                        }
                    }
                }
                gate_inputs.push((g_idx, inputs));
            }
        }
    }

    // Now construct nets based on driver-receiver relationships
    // A net consists of its driver and all its receivers
    let mut net_receivers: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (receiver_idx, in_nets) in gate_inputs {
        for net_name in in_nets {
            net_receivers.entry(net_name).or_default().push(receiver_idx);
        }
    }

    for (net_name, driver_idx) in net_to_driver {
        let mut connected_gates = vec![driver_idx];
        if let Some(receivers) = net_receivers.remove(&net_name) {
            connected_gates.extend(receivers);
        }

        if connected_gates.len() > 1 {
            circuit.add_net(&net_name, connected_gates, vec![]);
        }
    }

    // Add remaining inputs that might be driven by primary inputs (not tracked in net_to_driver)
    for (net_name, receivers) in net_receivers {
        if receivers.len() > 1 {
            // Treat as a net driven by a primary input
            circuit.add_net(&net_name, receivers, vec![]);
        } else if receivers.len() == 1 {
            // Warn if strict checking is needed, but safe to ignore single-pin nets
            // warn!("Net {} is floating or connected to primary input only", net_name);
        }
    }

    circuit.build_adjacency();
    info!("Parsed BLIF: {} gates, {} nets", circuit.num_gates, circuit.num_nets);

    Ok(circuit)
}
