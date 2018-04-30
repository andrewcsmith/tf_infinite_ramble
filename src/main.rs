#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate tensorflow;
extern crate hound;
extern crate sample;

use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::process::exit;

use sample::Sample;
use tensorflow::{Code, Graph, ImportGraphDefOptions, Session, SessionOptions, Status, StepWithGraph, Tensor};

/// These should probably be in the tensorflow graph somehow
static HOP: usize = 2048;
static BIN: usize = 2048;

fn main() {
    exit(match go() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}",e );
            1
        }
    })
}

fn go() -> Result<(), Box<Error>> {
    let filename = "./model.pb";
    let source = read_wav("sources/full_source.wav")?;
    let target = read_wav("sources/target.wav")?;

    let source_tensor = Tensor::new(&[source.len() as u64])
        .with_values(&source)?;
    let target_tensor = Tensor::new(&[target.len() as u64])
        .with_values(&target)?;

    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?
        .read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;

    // Run the Step
    let mut session = Session::new(&SessionOptions::new(), &graph)?;
    let mut step = StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("source_pcm")?, 0, &source_tensor);
    step.add_input(&graph.operation_by_name_required("target_pcm")?, 0, &target_tensor);
    let start_frames = step.request_output(&graph.operation_by_name_required("start_frames")?, 0);
    session.run(&mut step)?;

    // This is also a thing we have to Just Know
    let res: Tensor<i64> = step.take_output(start_frames)?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("out.wav", spec)?;

    for start_frame in res.iter() {
        let samps = &source[(*start_frame as usize)..(*start_frame as usize+BIN)];
        for s in samps {
            writer.write_sample(s.to_sample::<i16>())?;
        }
    }

    writer.finalize()?;

    Ok(())
}

fn read_wav(path: &str) -> Result<Vec<f32>, Box<Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let samples = reader.samples::<i16>()
        .map(|s| s.unwrap().to_sample())
        .collect();
    Ok(samples)
}
