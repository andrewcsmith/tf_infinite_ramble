# Tensorflow infinite_ramble

I've been working on a project lately that's a very basic MFCC matching
program, because MFCC calculation seems to be one of the major things that's
slowing down much of my Rust audio processing code. Given that there's quite
a bit of room for improvement, I've used this project as a way to try out
different efficient computational methods. 

The following file uses tensorflow to compute the cosine similarity matrices
of the MFCCs of two sound files, calculate the angular distance (using acos),
and find the minimum value, which would imply that the two frames are
maximally correlated. It uses frames from a "source" file to reconstruct an
entire "target" file. Using it with two speech samples, recorded on the same
microphone, provides a fairly reasonable way of testing by ear, but using
samples of totally different sources is cool too.

## How do

You need the rust bindings for tensorflow working and installed, as well as
the usual python3 tensorflow. Make sure you can pass the tests in
[tensorflow/rust](https://github.com/tensorflow/rust) before attempting this
project.

I'm running this on Windows, which is a major headache. Don't do that.

```
python similarity_matrix.py # => outputs model.pb
cargo run --release
```
