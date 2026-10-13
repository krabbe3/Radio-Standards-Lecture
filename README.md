# Radio Standards

Lecture materials for the course **Radio Standards** at the Technical University of Ilmenau.

## Course Information

- **Instructors:** Lorenz Mohr, Dr.-Ing. Alexander Ihlow, Prof. Thomas Dallmann
- **Institute:** Fachgebiet Elektronische Messtechnik und Signalverarbeitung (EMS)

## Course Overview

This course provides a comprehensive introduction to radio communication standards, their technical foundations, and the standardization process. Students will learn how to read, understand, and work with technical standards used in modern wireless systems.

### Topics

- Technical preliminaries (LTI channels, Fourier analysis, spatial processing)
- Radio wave propagation
- Standardization fundamentals (spectrum allocation, patents, standardization bodies)
- Evolution of cellular standards (0G through 5G)
- Deep dive into OFDM
- Comparison of WiFi, Bluetooth, and cellular technologies
- DVB standards (practical standard reading)
- Business models and commercialization of standards
- Future trends: 6G and beyond

## Repository Structure

```
radio-standards/
├── presentation/       # Beamer presentation slides (LaTeX)
│   ├── RS.tex          # Main presentation file
│   ├── sections/       # Individual lecture sections
│   ├── images/         # Figures and diagrams
│   └── references.bib  # Bibliography
├── scripts/            # Python demonstration scripts
│   └── section-0/      # Introductory demos (beamforming, spectrograms)
```

## Prerequisites

- Signal and system theory
- Basic engineering mathematics

## Building the Presentation

Install LaTeX with Beamer support, then compile:

```bash
cd presentation
pdflatex RS
```

## Python Environment

### Install Miniconda

If you don't have Conda installed, download and install Miniconda:

```bash
# Download installer (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer (follow prompts, accept defaults)
bash Miniconda3-latest-Linux-x86_64.sh

# Re-source shell or log out/in
source ~/.bashrc
```

For macOS and Windows, see the official installer: https://docs.conda.io/en/latest/miniconda.html

### Create the Environment

```bash
conda create -n radio-standards python=3.11 -y
conda activate radio-standards
conda install numpy matplotlib -y
pip install sounddevice
```

On Linux, `sounddevice` requires the PortAudio library. Install it via your package manager (e.g., `sudo apt install libportaudio2`).

## Demo Scripts

The `scripts/section-0/` directory contains interactive Python demonstrations:

- `live-spectrogram.py` - Real-time FFT visualization from microphone input
- `bartlett.py` - Bartlett beamformer visualization
- `bartlett-multi-doa.py` - Multiple direction-of-arrival estimation
- `rect_sinc.py` - Rectangular pulse and sinc function demonstration

